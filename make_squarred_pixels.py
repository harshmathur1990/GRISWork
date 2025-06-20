from scipy.interpolate import RectBivariateSpline
import sunpy.io
import numpy as np
from pathlib import Path

# Constants
pixel_scale_desired = 0.135  # arcsec/pixel
roi_y = 3 * 2                # arcsec
roi_x = 5.94 * 2             # arcsec

def reinterpolate_to_square_pixel_scale(base_path, filename):
    data, _ = sunpy.io.read_file(base_path / filename)[0]

    if len(data.shape) == 5:
        reshaped_data = np.transpose(data, axes=(0, 1, 4, 2, 3))  # (T, P, Y, X, W) → (T, P, W, Y, X)
    else:
        
        reshaped_data = np.transpose(data, axes=(0, 3, 1, 2))  # (P, Y, X, W) → (P, W, Y, X)

    # Determine current pixel scales
    pixel_scale_y = roi_y / reshaped_data.shape[-2]
    pixel_scale_x = roi_x / reshaped_data.shape[-1]

    # New grid
    n_pix_y = int(roi_y / pixel_scale_desired)
    n_pix_x = int(roi_x / pixel_scale_desired)

    y = np.arange(0, reshaped_data.shape[-2]) * pixel_scale_y
    x = np.arange(0, reshaped_data.shape[-1]) * pixel_scale_x
    new_y = np.arange(0, n_pix_y) * pixel_scale_desired
    new_x = np.arange(0, n_pix_x) * pixel_scale_desired

    def interp_image(image2d):
        interp = RectBivariateSpline(y, x, image2d, kx=1, ky=1)
        return interp(new_y, new_x)

    # Interpolate each slice
    new_data = np.empty(reshaped_data.shape[:-2] + (n_pix_y, n_pix_x))
    it = np.ndindex(reshaped_data.shape[:-2])
    for idx in it:
        new_data[idx] = interp_image(reshaped_data[idx])

    # Reverse reshape
    if len(data.shape) == 5:
        new_reshaped_data = np.transpose(new_data, axes=(0, 1, 3, 4, 2))  # (T, P, Y, X, W)
        new_reshaped_data = new_reshaped_data[:, :, ::-1, ::-1, :]        # Flip if needed
    else:
        new_reshaped_data = np.transpose(new_data, axes=(0, 2, 3, 1))     # (T, Y, X, W)
        new_reshaped_data = new_reshaped_data[:, ::-1, ::-1, :]           # Flip if needed

    # Save output
    sunpy.io.write_file(
        base_path / f'{filename}_squarred_pixels.fits',
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
