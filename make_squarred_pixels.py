from scipy.interpolate import RectBivariateSpline
import sunpy.io
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


pixel_scale_desired = 0.135

roi_y = 3 * 2

roi_x = 5.94 * 2


def prepare_interp(x, y, new_x, new_y):
    def reinterp(image):
        interp = RectBivariateSpline(x, y, image)
        return interp(new_x, new_y)

	return reinterp


def reinterpolate_to_square_pixel_scale(base_path, filename):
    data, _ = sunpy.io.read_file(base_path / filename)[0]

    if len(data.shape) == 5:
        reshaped_data = np.transpose(data, axes=(0, 1, 4, 2, 3))
    else:
        reshaped_data = np.transpose(data, axes=(0, 3, 1, 2))

    pixel_scale_y = roi_y / reshaped_data.shape[-2]

    pixel_scale_x = roi_x / reshaped_data.shape[-1]

    n_pix_y = int(roi_y / pixel_scale_desired)

    n_pix_x = int(roi_x / pixel_scale_desired)

    y = np.arange(0, reshaped_data.shape[-2] * pixel_scale_y, pixel_scale_y)

    x = np.arange(0, reshaped_data.shape[-1] * pixel_scale_x, pixel_scale_x)

    new_y = np.arange(0, n_pix_y * pixel_scale_desired, pixel_scale_desired)

    new_x = np.arange(0, n_pix_x * pixel_scale_desired, pixel_scale_desired)

    reinterp = prepare_interp(y, x, new_y, new_x)

    vec_reinterp = np.vectorize(reinterp, signature='(m,n)->(a,b)')

    new_data = vec_reinterp(reshaped_data)

    if len(data.shape) == 5:
        new_reshaped_data = np.transpose(new_data, axes=(0, 1, 3, 4, 2))
    else:
        new_reshaped_data = np.transpose(new_data, axes=(0, 2, 3, 1))

    sunpy.io.write_file(
        base_path / '{}_squarred_pixels.fits'.format(filename),
        new_reshaped_data,
        dict(),
        overwrite=True
    )


if __name__ == '__main__':
    base_path = Path('/mnt/f/GRIS')

    filenames = [
        '25Apr25ARM1-003.fits',
        '25Apr25ARM1-004.fits',
        '25Apr25ARM2-003.fits',
        '25Apr25ARM2-004.fits'
    ]

    for filename in filenames:
        reinterpolate_to_square_pixel_scale(base_path, filename)
