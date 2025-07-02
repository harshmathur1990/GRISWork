import numpy as np
import sunpy.io
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import scipy.ndimage
from copy import deepcopy
from spectral_veil_library import normalise_profiles, approximate_spectral_veil_and_sigma, correct_for_spectral_veil


def calculate_spectral_veil_from_median_profile(
    fits_file,
    falc_output,
    write_path,
    cont_wave_index,
    wave,
    wavelength
):

    data, header = sunpy.io.read_file(fits_file)[0]

    if len(data.shape) == 5:
        image = data[0, 0, :, :, cont_wave_index]
    else:
        image = data[0, :, :, cont_wave_index]

    plt.imshow(image, origin='lower', cmap='gray')
    points = np.array(plt.ginput(2, 600))

    x1, y1 = points[0]
    x2, y2 = points[1]

    # Determine bounds
    x_min, x_max = sorted([int(round(x1)), int(round(x2))])
    y_min, y_max = sorted([int(round(y1)), int(round(y2))])

    if len(data.shape) == 5:
        median_profile = np.median(data[:, 0, y_min:y_max+1, x_min:x_max+1], axis=(0, 1, 2))
    else:
        median_profile = np.median(data[0, y_min:y_max+1, x_min:x_max+1], axis=(0, 1))

    fal = h5py.File(falc_output, 'r')

    norm_line, norm_atlas, atlas_wave = normalise_profiles(
        median_profile,
        wave,
        fal['profiles'][0, 0, 0, :, 0],
        fal['wav'][()],
        cont_wave=wave[cont_wave_index]
    )

    fac_ind_s = 0
    fac_ind_e = norm_atlas.size - 1

    result, result_atlas, fwhm, sigma, k_values = approximate_spectral_veil_and_sigma(
        norm_line[fac_ind_s:fac_ind_e],
        norm_atlas[fac_ind_s:fac_ind_e],
    )

    r_fwhm = fwhm[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_sigma = sigma[np.unravel_index(np.argmin(result), result.shape)[0]]

    r_spectral_veil = k_values[np.unravel_index(np.argmin(result), result.shape)[1]]

    broadening_km_sec = np.abs(r_fwhm * (wave[1] - wave[0]) * 2.99792458e5 / wavelength)

    veil_corrected_median = median_profile

    veil_corrected_median = correct_for_spectral_veil(
        median_profile,
        r_spectral_veil,
        cont_wave_index
    )

    ind_synth = np.argmin(np.abs(fal['wav'][()] - wave[cont_wave_index]))

    cgs_calib_factor = veil_corrected_median[cont_wave_index] / fal['profiles'][0, 0, 0, ind_synth, 0]

    f = h5py.File(
        write_path / 'spectral_veil_estimated_profile_{}.h5'.format(fits_file.name), 'w')

    f['wave'] = wave

    f['norm_atlas'] = norm_atlas

    f['median_profile'] = median_profile

    f['norm_median'] = norm_line

    f['mean_square_error'] = result

    f['result_atlas'] = result_atlas

    f['fwhm_in_pixels'] = r_fwhm

    f['sigma_in_pixels'] = r_sigma

    f['spectral_veil_value'] = r_spectral_veil

    f['fwhm'] = fwhm

    f['sigma'] = sigma

    f['k_values'] = k_values

    f['broadening_in_km_sec'] = broadening_km_sec

    f['cgs_calib_factor'] = cgs_calib_factor

    f.close()

    norm_median_veil, norm_atlas, atlas_wave = normalise_profiles(
        veil_corrected_median,
        wave,
        fal['profiles'][0, 0, 0, :, 0],
        fal['wav'][()],
        cont_wave=wave[cont_wave_index]
    )

    plt.close('all')

    plt.plot(wave, norm_median_veil, label='veil Corrected Median')

    plt.plot(wave, scipy.ndimage.gaussian_filter1d(norm_atlas, sigma=r_sigma),
             label='Atlas')

    plt.legend()

    plt.savefig(write_path / '{}_median_comparison.pdf'.format(fits_file.name), format='pdf', dpi=600)

    plt.show()

    return r_fwhm, r_sigma, r_spectral_veil, wave, wavelength, broadening_km_sec, norm_median_veil, cgs_calib_factor


def do_spectral_veil_correction_main_func():

    base_path = Path(
        '/mnt/f/GRIS'
    )

    falc_output = base_path / 'FALC_GRIS_IFU_0p79.nc'

    fits_data = [
        (
            'CA',
            [
                base_path / '25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits',
                base_path / '25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
            ]
        ),
        (
            'HE',
            [
                base_path / '25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits',
                base_path / '25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
            ]
        ),
    ]

    wave_CA = np.arange(8540.67304823, 8540.67304823 + 1000 * 0.0109907, 0.0109907)[0:1000]

    wave_HE = np.arange(10818.6544101, 10818.6544101 + 872 * 0.0144423, 0.0144423)[0:872]

    wavelength = 8542

    cont_wave_index = -1

    vec_correct_for_spectral_veil = np.vectorize(
        correct_for_spectral_veil, signature='(n),(),()->(n)'
    )

    for atom, fits_files in fits_data:

        if atom == 'CA':
            wave = wave_CA
            wavelength = 8542
            cont_wave_index = -1
        else:
            wave = wave_HE
            wavelength = 10827
            cont_wave_index = 0

        a, b, c, d, e, f, g, h = calculate_spectral_veil_from_median_profile(
            fits_file=fits_files[0],
            falc_output=falc_output,
            write_path=base_path,
            cont_wave_index=cont_wave_index,
            wave=wave,
            wavelength=wavelength
        )

        r_fwhm, r_sigma, r_spectral_veil, wave, wavelength, broadening_km_sec, norm_median_veil, cgs_calib_factor = a, b, c, d, e, f, g, h

        for fits_file in fits_files:
            print (fits_file.name)
            data, header = sunpy.io.read_file(fits_file)[0]

            if len(data.shape) == 5:
                intensity = data[:, 0]
            else:
                intensity = data[0]

            new_data = deepcopy(data)

            veil_corrected_data = vec_correct_for_spectral_veil(
                intensity,
                r_spectral_veil,
                cont_wave_index,
            )

            if len(data.shape) == 5:
                new_data[:, 0] = veil_corrected_data
            else:
                new_data[0] = veil_corrected_data

            new_data = new_data / cgs_calib_factor

            sunpy.io.write_file(
                base_path / 'spectralveil_corrected_{}'.format(fits_file.name),
                new_data,
                header,
                overwrite=True
            )


if __name__ == '__main__':
    do_spectral_veil_correction_main_func()
