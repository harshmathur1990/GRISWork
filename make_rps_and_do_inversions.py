import sys
sys.path.insert(1, '/mn/stornext/u3/harshm/Documents/WorkRepo/stic/example')
import numpy as np
import h5py
import netCDF4 as nc
import sunpy.io
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import matplotlib.gridspec as gridspec
from prepare_data import *
import scipy.ndimage
from tqdm import tqdm


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')

kmeans_output_dir = base_path / 'K-Means-PCA'
input_files = [
    base_path / 'spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits',
    base_path / 'spectralveil_corrected_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
]

input_files_silicon = [
    base_path / 'spectralveil_corrected_25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits',
    base_path / 'spectralveil_corrected_25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'
]

kmeans_file = base_path / 'chosen_PCA_out_SV_100.h5'

falc_file = base_path / 'FALC.nc'

rps_plot_write_dir = base_path / 'PCA_RPs_Plots'


def get_ltau_scale():
    taumin = -7.8
    taumax= 1.0
    dtau = 0.14
    ntau = int((taumax-taumin)/dtau) + 1
    ltau_scale = np.arange(ntau, dtype='float64')/(ntau-1.0) * (taumax-taumin) + taumin

    return ltau_scale


def make_rps():
    f = h5py.File(kmeans_file, 'r+')

    framerows = None

    t1, y1, x1, y2, x2 = 0, 0, 0, 0, 0

    for input_file in input_files:
        data, header = sunpy.io.read_file(input_file)[0]

        if len(data.shape) == 5:
            data = np.transpose(data, axes=(0, 2, 3, 1, 4))

            t1, y1, x1 = data.shape[0], data.shape[1], data.shape[2]

            data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
        else:
            data = np.transpose(data, axes=(1, 2, 0, 3))

            y2, x2 = data.shape[0], data.shape[1]

            data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

        if framerows is None:
            framerows = data

        else:

            framerows = np.concatenate([framerows, data], axis=0)

    keys = ['rps', 'final_labels_1', 'final_labels_2']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()]

    labels_1 = f['labels_'][0:t1 * y1 * x1].reshape((t1, y1, x1)).astype(np.int8)

    labels_2 = f['labels_'][t1 * y1 * x1:].reshape((y2, x2)).astype(np.int8)

    f['final_labels_1'] = labels_1

    f['final_labels_2'] = labels_2

    total_labels = max(labels_1.max(), labels_2.max()) + 1

    rps = np.zeros(
        (total_labels, framerows.shape[2], 4),
        dtype=np.float64
    )

    for i in range(total_labels):
        a = np.where(labels == i)[0]
        rps[i, :, 0] = np.mean(framerows[a, 0], axis=0)
        rps[i, :, 1] = np.mean(framerows[a, 1] / framerows[a, 0], axis=0) * rps[i, :, 0]
        rps[i, :, 2] = np.mean(framerows[a, 2] / framerows[a, 0], axis=0) * rps[i, :, 0]
        rps[i, :, 3] = np.mean(framerows[a, 3] / framerows[a, 0], axis=0) * rps[i, :, 0]

    f['rps'] = rps

    f.close()


def make_si_rps():
    f = h5py.File(kmeans_file, 'r+')

    framerows = None

    t1, y1, x1, y2, x2 = 0, 0, 0, 0, 0

    for input_file in input_files_silicon:
        data, header = sunpy.io.read_file(input_file)[0]

        if len(data.shape) == 5:
            data = np.transpose(data, axes=(0, 2, 3, 1, 4))

            t1, y1, x1 = data.shape[0], data.shape[1], data.shape[2]

            data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
        else:
            data = np.transpose(data, axes=(1, 2, 0, 3))

            y2, x2 = data.shape[0], data.shape[1]

            data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

        if framerows is None:
            framerows = data

        else:

            framerows = np.concatenate([framerows, data], axis=0)

    keys = ['rps_silicon']

    for key in keys:
        if key in list(f.keys()):
            del f[key]

    labels = f['labels_'][()]

    total_labels = labels.max() + 1

    rps = np.zeros(
        (total_labels, framerows.shape[2], 4),
        dtype=np.float64
    )

    for i in range(total_labels):
        a = np.where(labels == i)[0]
        rps[i, :, 0] = np.mean(framerows[a, 0], axis=0)
        rps[i, :, 1] = np.mean(framerows[a, 1] / framerows[a, 0], axis=0) * rps[i, :, 0]
        rps[i, :, 2] = np.mean(framerows[a, 2] / framerows[a, 0], axis=0) * rps[i, :, 0]
        rps[i, :, 3] = np.mean(framerows[a, 3] / framerows[a, 0], axis=0) * rps[i, :, 0]

    f['rps_silicon'] = rps

    f.close()


def get_farthest(whole_data, a, center, r):
    all_profiles = whole_data[a, :, r]
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index], index


def get_closest(whole_data, a, center, r):
    all_profiles = whole_data[a, :, r]
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[0]
    return all_profiles[index], index


def get_farthest_atmosphere(whole_data, a, center):
    all_profiles = whole_data[a, :]
    difference = np.sqrt(
        np.sum(
            np.square(
                np.subtract(
                    all_profiles,
                    center
                )
            ),
            axis=1
        )
    )
    index = np.argsort(difference)[-1]
    return all_profiles[index]


def get_max_min(whole_data, a, r):
    all_profiles = whole_data[a, :, r]
    return all_profiles.max(), all_profiles.min()


def get_data(get_data=True, get_labels=True, get_rps=True, crop_indice=None, cw=8542):
    whole_data, labels, rps = None, None, None

    if int(cw) == 8542:
        wave = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823
    else:
        wave = np.arange(872, dtype=float) * 0.0144423 + 10818.6544101

    if get_data:
        framerows = None

        for input_file in input_files_silicon:
            data, header = sunpy.io.read_file(input_file)[0]

            if len(data.shape) == 5:
                data = np.transpose(data, axes=(0, 2, 3, 4, 1))

                data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3], data.shape[4]))
            else:
                data = np.transpose(data, axes=(1, 2, 3, 0))

                data = data.reshape((data.shape[0] * data.shape[1], data.shape[2], data.shape[3]))

            if framerows is None:
                framerows = data

            else:

                framerows = np.concatenate([framerows, data], axis=0)

        framerows[:, :, 1:4] = framerows[:, :, 1:4] / framerows[:, :, 0][:, :, np.newaxis]

        whole_data = framerows

    f = h5py.File(kmeans_file, 'r')

    if get_labels:

        labels = f['labels_'][()]
        # if crop_indice is not None:
        #     labels = labels[crop_indice[0][1]:crop_indice[1][1], crop_indice[0][0]:crop_indice[1][0]]
        # labels = labels.reshape(labels.shape[0] * labels.shape[1])

    if get_rps:
        if int(cw) == 8542:
            rps = f['rps'][()]
        else:
            rps = f['rps_silicon'][()]

        # rps /= cont[0]
        rps[:, :, 1:4] /= rps[:, :, 0][:, :, np.newaxis]

    f.close()

    return whole_data, labels, rps, wave


def make_rps_plots(name='RPs', cw=8542.09):
    whole_data, labels, rps, wave = get_data(crop_indice=None, cw=cw)

    k = 0

    color = 'black'

    cm = 'Blues'

    wave_x = np.arange(wave.size)

    xticks = list()

    xticks.append(np.argmin(np.abs(wave - cw)))

    for m in range(5):

        plt.close('all')

        plt.clf()

        plt.cla()

        fig = plt.figure(figsize=(8.27, 11.69))

        subfigs = fig.subfigures(5, 4)

        for i in range(5):

            for j in range(4):

                gs = gridspec.GridSpec(2, 2)

                gs.update(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

                r = 0

                # sys.stdout.write('{}\n'.format(k))

                subfig = subfigs[i][j]

                a = np.where(labels == k)[0]

                for p in range(2):
                    for q in range(2):

                        ax1 = subfig.add_subplot(gs[r])

                        center = rps[k, :, r]

                        farthest_profile, index_far = get_farthest(whole_data, a, center, r)

                        ax1.plot(
                            wave_x,
                            center,
                            color=color,
                            linewidth=0.25,
                            linestyle='-'
                        )

                        # if a.size > 0:
                        c, f = get_max_min(whole_data, a, r)

                        max_8542, min_8542 = c, f

                        # if r == 1:
                        #     max_8542, min_8542 = 0.06, -0.06
                        # else:
                        #     max_8542, min_8542 = 1, 0
                        # else:
                        min_8542 = min_8542 * 0.9
                        max_8542 = max_8542 * 1.1

                        in_bins_8542 = np.linspace(min_8542, max_8542, wave.shape[0])

                        H1, xedge1, yedge1 = np.histogram2d(
                            np.tile(wave_x, a.shape[0]),
                            whole_data[a, :, r].flatten(),
                            bins=(wave_x, in_bins_8542)
                        )

                        ax1.plot(
                            wave_x,
                            farthest_profile,
                            color=color,
                            linewidth=0.5,
                            linestyle='dotted'
                        )

                        ymesh = H1.T

                        # ymeshmax = np.max(ymesh, axis=0)

                        ymeshnorm = ymesh / ymesh.max()

                        X1, Y1 = np.meshgrid(xedge1, yedge1)

                        ax1.pcolormesh(X1, Y1, ymeshnorm, cmap=cm)

                        # else:
                        #     max_8542, min_8542 = np.min(center), np.max(center)
                        #     min_8542 = min_8542 * 0.9
                        #     max_8542 = max_8542 * 1.1

                        ax1.set_ylim(min_8542, max_8542)

                        if r == 0:
                            ax1.text(
                                0.2,
                                0.6,
                                'n = {}'.format(
                                    a.size
                                ),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                            ax1.text(
                                0.3,
                                0.8,
                                'RP {}'.format(k),
                                transform=ax1.transAxes,
                                fontsize=8
                            )

                        ax1.set_xticks(xticks)
                        ax1.set_xticklabels([])

                        if r == 0:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    2
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    2
                                )
                            ]
                        else:
                            y_ticks = [
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.1,
                                    4
                                ),
                                np.round(
                                    min_8542 + (max_8542 - min_8542) * 0.8,
                                    4
                                )
                            ]

                        ax1.set_yticks(y_ticks)
                        ax1.set_yticklabels(y_ticks)

                        ax1.tick_params(axis="y", direction="in", pad=-30)

                        r += 1

                k += 1

        fig.savefig(
            rps_plot_write_dir / '{}_{}.png'.format(name, k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()



def make_paper_rps_plots(name='RPs', cw=8542.09):
    whole_data, labels, rps, wave = get_data(crop_indice=None)

    whole_data[:, :, 1:4] *= 100

    color = 'black'

    cm = 'Blues'

    wave_x = wave

    xticks = [cw]

    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 6}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(15, 7, figsize=(7, 7))

    r = 1

    k = 0

    for i in range(15):

        for j in range(7):

            a = np.where(labels == k)[0]

            center = rps[k, :, r] * 100

            farthest_profile, index_far = get_farthest(whole_data, a, center, r)

            closest_profile, index_close = get_closest(whole_data, a, center, r)

            nx = 504

            ny = 504

            step = 4

            x, y = (
                np.repeat(np.arange(0, nx, step)[:, np.newaxis], repeats=nx // step, axis=1).flatten(),
                np.repeat(np.arange(0, ny, step)[:, np.newaxis], repeats=ny // step, axis=1).T.flatten()
            )

            max_8542, min_8542 = get_max_min(whole_data, a, r)

            max_8542 = np.ceil(max_8542)

            min_8542 = np.floor(min_8542)

            in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

            H1, xedge1, yedge1 = np.histogram2d(
                np.tile(wave_x, a.shape[0]),
                whole_data[a, :, r].flatten(),
                bins=(wave_x, in_bins_8542)
            )

            ymesh = H1.T

            ymeshnorm = ymesh / ymesh.max()

            X1, Y1 = np.meshgrid(xedge1[:-1], yedge1[:-1])

            axs[i][j].plot(
                wave_x,
                center,
                color='brown',
                linewidth=0.25,
                linestyle='-'
            )

            axs[i][j].plot(
                wave_x,
                farthest_profile,
                color='brown',
                linewidth=0.5,
                linestyle='--'
            )

            axs[i][j].axvline(
                4226.73,
                color='black',
                linewidth=0.25,
                linestyle='--'
            )

            axs[i][j].pcolormesh(X1, Y1, ymeshnorm, shading='gouraud', cmap=cm, rasterized=True, vmin=0, vmax=0.05)

            axs[i][j].set_ylim(-7.1, 7.1)

            axs[i][j].text(
                0.6,
                0.9,
                '{} %'.format(
                    np.round(a.size * 100 / 15876, 2)
                ),
                transform=axs[i][j].transAxes
            )

            axs[i][j].text(
                0.2,
                0.9,
                '# {}'.format(k),
                transform=axs[i][j].transAxes
            )

            axs[i][j].set_xticks([])
            axs[i][j].set_xticklabels([])

            y_ticks = [
                -4, 0, 4
            ]

            axs[i][j].set_yticks(y_ticks)
            axs[i][j].set_yticklabels(y_ticks)

            k += 1

            if not (i == 14 and j == 0):
                axs[i][j].axis('off')
            else:
                axs[i][j].spines[['right', 'top']].set_visible(False)
                axs[i][j].set_xticks([4226, 4227, 4228])
                axs[i][j].set_xticklabels([4226, 4227, 4228])
                axs[i][j].text(
                    0.15,
                    -0.68,
                    r'Wavelength [$\mathrm{\AA}$]',
                    transform=axs[i][j].transAxes
                )
                axs[i][j].text(
                    -0.35,
                    -0.05,
                    r'Stokes $Q/I$ [%]',
                    transform=axs[i][j].transAxes,
                    rotation=90
                )


    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.98, wspace=0.05, hspace=0.01)

    fig.savefig(
        '/home/harsh/CourseworkRepo/rh-rene/figures/RPs.pdf',
        format='pdf',
        dpi=300
    )

    plt.close('all')

    plt.clf()

    plt.cla()


def make_observation_object(
    write_path, rps,
    wave_name, wave,
    rps_profiles, core_indice,
    r_sigma,
    cont,
    factor=4,
    continuum_correction=1,
    all_weight=0.004,
    core_weight=0.001,
    ignore_indice=None,
    create_sigma_files=False
):
    wc, ic = findgrid(wave, (wave[1] - wave[0]) / factor, extra=8)

    obs = sp.profile(nx=rps.size, ny=1, ns=4, nw=wc.size)

    obs.wav[:] = wc[:]

    obs.dat[0, 0, :, ic, :] = np.transpose(
        rps_profiles[rps],
        axes=(1, 0, 2)
    ) * continuum_correction

    obs.weights[:, :] = 1.e16  # Very high value means weight zero
    obs.weights[ic, 0] = all_weight
    obs.weights[ic[core_indice[0]:core_indice[1]], 0] = core_weight

    if ignore_indice is not None:
        obs.weights[ic[ignore_indice[0]:ignore_indice[1]], 0] = 1e16

    formatted_string = ''

    if create_sigma_files:
        if wave.size%2 == 0:
            kernel_size = wave.size - 1
        else:
            kernel_size = wave.size - 2

        rev_kernel = np.zeros(kernel_size)
        rev_kernel[kernel_size//2] = 1
        kernel = scipy.ndimage.gaussian_filter1d(rev_kernel, sigma=r_sigma * factor)

        broadening_filename = 'gaussian_broadening_{}_pixel_{}.h5'.format(r_sigma * factor, wave_name)

        f = h5py.File(write_path / broadening_filename, 'w')
        f['iprof'] = kernel
        f['wav'] = np.zeros_like(kernel)
        f.close()

        lab = "region = {0:10.5f}, {1:8.5f}, {2:3d}, {3:e}, {4}"

        formatted_string = lab.format(
            obs.wav[0],
            obs.wav[1] - obs.wav[0],
            obs.wav.shape[0],
            cont,
            'spectral, {}'.format(broadening_filename)
        )

    return formatted_string, obs


def make_stic_inversion_files_si_ca_rps(rps=None, get_region=False):
    si_core_indice = [400, 656]

    ca2_core_indice = [0, 226]

    si_ignore_indice = [656, 872]

    # si_ignore_indice = [0, 872]

    wave_names = ['SiI_10827', 'CaII_8542']

    f = h5py.File(kmeans_file, 'r')

    fcastray = h5py.File(base_path / 'spectral_veil_estimated_profile_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits.h5', 'r')

    fsistray = h5py.File(base_path / 'spectral_veil_estimated_profile_25Apr25ARM1-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits.h5', 'r')

    ca_rps = f['rps'][()]

    si_rps = f['rps_silicon'][()]

    r_sigma_ca = fcastray['sigma_in_pixels'][()]

    r_sigma_si = fsistray['sigma_in_pixels'][()]

    fcastray.close()

    fsistray.close()

    if rps is None:
        rps = range(f['rps'].shape[0])

    rps = np.array(rps)

    wave_CA = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823

    wave_SI = np.arange(872, dtype=float) * 0.0144423 + 10818.6544101

    formatted_string_ca, obs_ca = make_observation_object(
        write_path=base_path, rps=rps,
        wave_name='CaII_8542', wave=wave_CA,
        rps_profiles=ca_rps, core_indice=ca2_core_indice,
        r_sigma=r_sigma_ca,
        cont=4.227725e-05,
        factor=4,
        continuum_correction=1,
        all_weight=0.004,
        core_weight=0.0005
    )

    formatted_string_si, obs_si = make_observation_object(
        write_path=base_path, rps=rps,
        wave_name='SiI_10827', wave=wave_SI,
        rps_profiles=si_rps, core_indice=si_core_indice,
        r_sigma=r_sigma_si,
        cont=4.0709165e-05,
        factor=4,
        continuum_correction=1,
        all_weight=0.004,
        core_weight=0.002,
        ignore_indice=si_ignore_indice
    )

    if get_region == True:
        print (formatted_string_ca)
        print (formatted_string_si)
        return

    if rps.size != f['rps'].shape[0]:
        writefilename = 'ca_si_rps_stic_profiles_x_{}_y_1.nc'.format('_'.join([str(_rp) for _rp in rps]))
    else:
        writefilename = 'ca_si_rps_stic_profiles_x_{}_y_1.nc'.format(rps.size)

    f.close()

    all_profiles = obs_ca + obs_si

    all_profiles.write(
        base_path / 'KMeans-Inversions' / writefilename
    )


def generate_input_atmos_file_from_falc(
        length,
        temp=None,
        vlos=None,
        in_file=None,
):
    if in_file is None:
        in_file = falc_file

    f_falc = h5py.File(in_file, 'r')

    m = sp.model(nx=length, ny=1, nt=1, ndep=f_falc['ltau500'][0, 0, 0].shape[0])

    m.ltau[:] = f_falc['ltau500'][0, 0, 0]
    m.temp[:] = f_falc['temp'][0, 0, 0]
    m.vturb[:] = f_falc['vturb'][0, 0, 0]
    m.vlos[:] = f_falc['vlos'][0, 0, 0]

    if temp is not None:
        m.temp[:] = np.interp(f_falc['ltau500'][0, 0, 0], temp[0], temp[1])

    if vlos is not None:
        m.vlos[:] = np.interp(f_falc['ltau500'][0, 0, 0], vlos[0], vlos[1])

    f_falc.close()

    m.pgas[:] = 1.0

    if temp is not None:
        m.write(base_path / 'KMeans-Inversions' / 'emission_{}.nc'.format(length))
        return

    m.write(base_path / 'KMeans-Inversions' / 'FALC_{}.nc'.format(length))


def make_individual_plots(
    write_path, finputprofs,
    fprofsresult, fatmosresult,
    index, rp,
    ind_8542,
    ind_10827
):
    plt.close('all')

    plt.clf()

    plt.cla()

    fig, axs = plt.subplots(4, 2, figsize=(7, 8))

    axs[0][0].plot(
        finputprofs['wav'][ind_8542] - 8542.09,
        finputprofs['profiles'][0, 0, index, ind_8542, 0],
        color='orange',
        linewidth=0.5
    )

    axs[0][0].plot(
        fprofsresult['wav'][ind_8542] - 8542.09,
        fprofsresult['profiles'][0, 0, index, ind_8542, 0],
        color='brown',
        linewidth=0.5
    )

    axs[0][0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    axs[0][1].plot(
        finputprofs['wav'][ind_8542] - 8542.09,
        finputprofs['profiles'][0, 0, index, ind_8542, 3] / finputprofs['profiles'][0, 0, index, ind_8542, 0],
        color='orange',
        linewidth=0.5
    )

    axs[0][1].plot(
        fprofsresult['wav'][ind_8542] - 8542.09,
        fprofsresult['profiles'][0, 0, index, ind_8542, 3] / fprofsresult['profiles'][0, 0, index, ind_8542, 0],
        color='brown',
        linewidth=0.5
    )

    axs[1][0].plot(
        finputprofs['wav'][ind_10827] - 10827.091,
        finputprofs['profiles'][0, 0, index, ind_10827, 0],
        color='orange',
        linewidth=0.5
    )

    axs[1][0].plot(
        fprofsresult['wav'][ind_10827] - 10827.091,
        fprofsresult['profiles'][0, 0, index, ind_10827, 0],
        color='brown',
        linewidth=0.5
    )

    axs[1][0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)

    axs[1][1].plot(
        finputprofs['wav'][ind_10827] - 10827.091,
        finputprofs['profiles'][0, 0, index, ind_10827, 3] / finputprofs['profiles'][0, 0, index, ind_10827, 0],
        color='orange',
        linewidth=0.5
    )

    axs[1][1].plot(
        fprofsresult['wav'][ind_10827] - 10827.091,
        fprofsresult['profiles'][0, 0, index, ind_10827, 3] / fprofsresult['profiles'][0, 0, index, ind_10827, 0],
        color='brown',
        linewidth=0.5
    )

    axs[2][0].plot(fatmosresult['ltau500'][0, 0, index], fatmosresult['temp'][0, 0, index] / 1e3, color='brown')

    axs[2][1].plot(fatmosresult['ltau500'][0, 0, index], fatmosresult['vlos'][0, 0, index] / 1e5, color='brown')

    axs[3][0].plot(fatmosresult['ltau500'][0, 0, index], fatmosresult['vturb'][0, 0, index] / 1e5, color='brown')

    axs[3][1].plot(fatmosresult['ltau500'][0, 0, index], fatmosresult['blong'][0, 0, index], color='brown')

    axs[0][0].set_xlabel(r'$\lambda(\AA)$')
    axs[0][0].set_ylabel(r'$I/I_{c}$')

    axs[0][1].set_xlabel(r'$\lambda(\AA)$')
    axs[0][1].set_ylabel(r'$V/I$')

    axs[1][0].set_xlabel(r'$\lambda(\AA)$')
    axs[1][0].set_ylabel(r'$I/I_{c}$')

    axs[1][1].set_xlabel(r'$\lambda(\AA)$')
    axs[1][1].set_ylabel(r'$V/I$')

    axs[2][0].set_xlabel(r'$\log(\tau_{500})$')
    axs[2][0].set_ylabel(r'$T[kK]$')

    axs[2][1].set_xlabel(r'$\log(\tau_{500})$')
    axs[2][1].set_ylabel(r'$V_{LOS}[Kms^{-1}]$')

    axs[3][0].set_xlabel(r'$\log(\tau_{500})$')
    axs[3][0].set_ylabel(r'$V_{turb}[Kms^{-1}]$')

    axs[3][1].set_xlabel(r'$\log(\tau_{500})$')
    axs[3][1].set_ylabel(r'$B_{long}[G]$')

    fig.tight_layout()

    fig.savefig(write_path / 'CA_SI_RPs_{}.pdf'.format(rp), format='pdf', dpi=300)

    plt.close('all')

    plt.clf()

    plt.cla()


def make_rps_inversion_plots(rps=None, indexes=None):

    if rps is None:
        rps = np.arange(100)
    else:
        rps = np.array(rps)

    common_rp_inversion_base = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions')

    inversion_path = Path('/mn/stornext/u3/harshm/Documents/WorkRepo/stic/run')

    # input_profile = common_rp_inversion_base / 'ca_si_rps_stic_profiles_x_100_y_1.nc'

    # output_profile = common_rp_inversion_base / 'ca_si_rps_stic_profiles_x_100_y_1_t_6_vlos_5_vturb_3_output_profs.nc'

    # output_atmos = common_rp_inversion_base / 'ca_si_rps_stic_profiles_x_100_y_1_t_6_vlos_5_vturb_3_output_atmos.nc'

    # input_profile = common_rp_inversion_base / 'Plots' / 'Plots_41' / 'ca_si_rps_stic_profiles_x_41_y_1.nc'

    # output_profile = common_rp_inversion_base / 'Plots' / 'Plots_41' / 'ca_si_rps_stic_profiles_x_41_y_1_t_7_vlos_7_vturb_4_output_profs.nc'

    # output_atmos = common_rp_inversion_base / 'Plots' / 'Plots_41' / 'ca_si_rps_stic_profiles_x_41_y_1_t_7_vlos_7_vturb_4_output_atmos.nc'

    input_profile = inversion_path / 'ca_si_rps_stic_profiles_x_98_y_1.nc'

    output_profile = inversion_path / 'ca_si_rps_stic_profiles_x_98_y_1_t_7_vlos_7_vturb_4_output_profs.nc'

    output_atmos = inversion_path / 'ca_si_rps_stic_profiles_x_98_y_1_t_7_vlos_7_vturb_4_output_atmos.nc'

    print (rps)

    print (input_profile)

    print (output_profile)

    print (output_atmos)

    finputprofs = h5py.File(input_profile, 'r')

    fprofsresult = h5py.File(output_profile, 'r')

    fatmosresult = h5py.File(output_atmos, 'r')

    nzind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    nzind_8542 = np.where(finputprofs['wav'][nzind] < 9000)[0]

    nzind_10827 = np.where(finputprofs['wav'][nzind] > 9000)[0]

    ind_8542 = nzind[nzind_8542]

    ind_10827 = nzind[nzind_10827]

    base_write_bath = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions/Plots')

    t = tqdm(total=rps.shape[0], desc="Generating plots", unit="RP")

    for index, rp in enumerate(rps):

        i_index = index

        if indexes is not None:
            i_index = indexes[index]

        write_path = base_write_bath / 'Plots_{}'.format(rp)

        write_path.mkdir(parents=True, exist_ok=True)

        make_individual_plots(
            write_path=write_path,
            finputprofs=finputprofs,
            fprofsresult=fprofsresult,
            fatmosresult=fatmosresult,
            index=i_index,
            rp=rp,
            ind_8542=ind_8542,
            ind_10827=ind_10827
        )

        t.update(1)

    finputprofs.close()

    fprofsresult.close()

    fatmosresult.close()


def calculate_continuum_correction():
    median_rps = np.array([93, 88, 47, 35, 18, 10, 5])

    median_rps.sort()

    inversion_path = Path('/mnt/d/Workrepo/stic/run')

    input_profile = inversion_path / 'ca_si_rps_stic_profiles_x_100_y_1.nc'

    output_profile = inversion_path / 'ca_si_rps_stic_profiles_x_100_y_1_t_6_vlos_3_vturb_1_output_profs.nc'

    finputprofs = h5py.File(input_profile, 'r')

    fprofsresult = h5py.File(output_profile, 'r')

    nzind = np.where(finputprofs['profiles'][0, 0, 0, :, 0] != 0)[0]

    nzind_8542 = np.where(finputprofs['wav'][nzind] < 9000)[0]

    nzind_10827 = np.where(finputprofs['wav'][nzind] > 9000)[0]

    ind_8542 = nzind[nzind_8542]

    ind_10827 = nzind[nzind_10827]

    median_profile_8542 = np.median(finputprofs['profiles'][0, 0, median_rps][:, ind_8542, 0], axis=0)

    median_profile_10827 = np.median(finputprofs['profiles'][0, 0, median_rps][:, ind_10827, 0], axis=0)

    synthesized_profile_8542 = np.median(fprofsresult['profiles'][0, 0, median_rps][:, ind_8542, 0], axis=0)

    synthesized_profile_10827 = np.median(fprofsresult['profiles'][0, 0, median_rps][:, ind_10827, 0], axis=0)

    plt.plot(median_profile_8542, color='black')

    plt.plot(synthesized_profile_8542, color='blue')

    plt.show()

    plt.plot(median_profile_10827, color='black')

    plt.plot(synthesized_profile_10827, color='blue')

    plt.show()

    correction_8542 = np.mean(synthesized_profile_8542[-20:]) / np.mean(median_profile_8542[-20:])

    correction_10827 = np.mean(synthesized_profile_10827[0:20]) / np.mean(median_profile_10827[0:20])

    print (np.round(correction_8542, 4))

    print (np.round(correction_10827, 4))

    finputprofs.close()

    fprofsresult.close()


def generate_input_atmos_file(
        length=30,
        temp=None,
        vlos=None,
        blong=0,
        name='',
        file=None,
        index=None,
        vlos_multiplier=None,
        use_doppler=False,
        doppler_ltau=None,
        rps=None
):

    if file is None:
        file = falc_file
        index = 0
        vlos_multiplier = 1
    elif index is None:
        index = 0
        vlos_multiplier = 1

    f = h5py.File(file, 'r')

    ltau_scale = get_ltau_scale()

    m = sp.model(nx=length, ny=1, nt=1, ndep=ltau_scale.shape[0])

    m.ltau[:, :, :] = ltau_scale

    m.pgas[:, :, :] = 1

    if not isinstance(index, list):

        m.temp[:, :, :] = np.interp(ltau_scale, f['ltau500'][0, 0, index], f['temp'][0, 0, index])

        m.vlos[:, :, :] = np.interp(ltau_scale, f['ltau500'][0, 0, index], f['vlos'][0, 0, index]) * vlos_multiplier

        m.vturb[:, :, :] = np.interp(ltau_scale, f['ltau500'][0, 0, index], f['vturb'][0, 0, index])

    else:
        for en_index, (ii, i_index, i_vlos_multiplier) in enumerate(zip(range(length), index, vlos_multiplier)):

            m.temp[0, 0, ii] = np.interp(ltau_scale, f['ltau500'][0, 0, i_index], f['temp'][0, 0, i_index])

            m.vturb[0, 0, ii] = np.interp(ltau_scale, f['ltau500'][0, 0, i_index], f['vturb'][0, 0, i_index])

            m.vlos[0, 0, ii] = np.interp(ltau_scale, f['ltau500'][0, 0, i_index], f['vlos'][0, 0, i_index]) * i_vlos_multiplier

    f.close()

    m.write(
        base_path / 'KMeans-Inversions' / 'atmos_{}_{}.nc'.format(length, name)
    )


def get_rp_atmos(rp_file):

    f = h5py.File(rp_file, 'r')

    temp = f['temp'][0, 0]
    vlos = f['vlos'][0, 0]
    vturb = f['vturb'][0, 0]

    f.close()

    return temp, vlos, vturb


def save_pixel_indices_for_rps(
    write_path,
    rps,
    rps_name,
    label_key_name='final_labels_1'
):
    """
    Save pixel indices for given representative profiles (RPs) based on the label map.

    Parameters
    ----------
    write_path : Path
        Directory where output HDF5 file will be written.
    rps : list or array
        List of representative profile labels to extract pixel indices for.
    rps_name : str
        Name identifier used in the output filename.
    label_key_name : str, optional
        Name of the dataset in the HDF5 file containing labels (default: 'final_labels_1').
    """

    # Read labels
    with h5py.File(kmeans_file, 'r') as f:
        labels = f[label_key_name][()]

    coord_lists = [[] for _ in range(labels.ndim)]  # a_list, b_list, (c_list if 3D)
    rp_values = []

    # Collect indices for each representative profile
    for rp in rps:
        indices = np.where(labels == rp)
        for dim, coord in enumerate(indices):
            coord_lists[dim].extend(coord.tolist())
        rp_values.extend([rp] * len(indices[0]))

    # Convert lists to numpy arrays
    coord_arrays = [np.array(coords, dtype=np.int64) for coords in coord_lists]
    rp_array = np.array(rp_values, dtype=np.int64)

    # Prepare final array: shape = (ndim + 1, N)
    num_dims = labels.ndim
    num_points = rp_array.size
    pixel_indices = np.zeros((num_dims + 1, num_points), dtype=np.int64)

    for i, arr in enumerate(coord_arrays):
        pixel_indices[i] = arr
    pixel_indices[num_dims] = rp_array

    # Output filename
    output_filename = write_path / f'pixel_indices_{rps_name}_total_{num_points}.h5'

    # Write to file
    with h5py.File(output_filename, 'w') as f_out:
        f_out['pixel_indices'] = pixel_indices

    print(f"✅ Saved pixel indices for {num_points} pixels → {output_filename}")

    np.savetxt(write_path / '{}.txt'.format(rps_name), rps)

    return pixel_indices


def combine_rps_atmos():
    kmeans_inversions = Path("/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions")
    base_path = kmeans_inversions / "Plots"
    hdf5_file = kmeans_inversions / "ca_si_rps_stic_profiles_x_100_y_1_t_6_vlos_5_vturb_3_output_atmos.nc"      # Path to your HDF5 file

    # --- Step 1: Load HDF5 datasets into local variables ---
    with h5py.File(hdf5_file, 'r') as f:
        temp = f['temp'][()]
        vlos = f['vlos'][()]
        vturb = f['vturb'][()]

    print(f"Loaded shapes -> temp: {temp.shape}, vlos: {vlos.shape}, vturb: {vturb.shape}")

    # --- Step 2: Iterate through folders Plos_0 to Plos_99 ---
    for i in range(100):
        folder = base_path / f"Plos_{i}"
        if not folder.is_dir():
            continue

        # Find file ending with 'output_atmos.nc'
        nc_files = list(folder.glob("*output_atmos.nc"))
        if not nc_files:
            continue

        nc_file = nc_files[0]

        try:
            # --- Step 3: Open NetCDF file and read variables ---
            with nc.Dataset(nc_file, 'r') as ds:
                if 'temp' not in ds.variables or 'vlos' not in ds.variables or 'vturb' not in ds.variables:
                    print(f"⚠️ Missing required variables in {nc_file}")
                    continue

                temp_val = ds['temp'][0, 0, 0]
                vlos_val = ds['vlos'][0, 0, 0]
                vturb_val = ds['vturb'][0, 0, 0]

            # --- Step 4: Replace corresponding entries in arrays ---
            temp[0, 0, i] = temp_val
            vlos[0, 0, i] = vlos_val
            vturb[0, 0, i] = vturb_val

            print(f"✅ Updated index {i} from {nc_file.name}")

        except Exception as e:
            print(f"❌ Error processing {nc_file}: {e}")

    # --- Step 5: Done ---
    print("\nAll folders processed. You can now save temp, vlos, vturb as needed.")

    ltau_scale = get_ltau_scale()

    m = sp.model(nx=100, ny=1, nt=1, ndep=ltau_scale.shape[0])

    m.ltau[:, :, :] = ltau_scale

    m.pgas[:, :, :] = 1

    m.temp[:] = temp

    m.vlos[:] = vlos

    m.vturb[:] = vturb

    m.write(kmeans_inversions / "combined_rps_atmos.nc")


def make_observation_object_caller(
    actual_filepath, pixel_indices,
    write_path,
    wave_name, wave,
    core_indice,
    r_sigma,
    cont,
    factor=4,
    continuum_correction=1,
    all_weight=0.004,
    core_weight=0.001,
    ignore_indice=None,
    create_sigma_files=False
):

    data, header = sunpy.io.read_file(actual_filepath)[0]

    if data.ndim == 5:
        seldata = data[pixel_indices[0], :, pixel_indices[1], pixel_indices[2]]
        seldata = np.transpose(seldata, axes=(0, 2, 1))
    else:
        seldata = data[:, pixel_indices[0], pixel_indices[1]]
        seldata = np.transpose(seldata, axes=(1, 2, 0))

    _, obs = make_observation_object(
        write_path=write_path,
        rps=np.arange(pixel_indices.shape[1]),
        wave_name=wave_name, wave=wave,
        rps_profiles=seldata, core_indice=core_indice,
        r_sigma=r_sigma,
        cont=cont,
        factor=factor,
        continuum_correction=continuum_correction,
        all_weight=all_weight,
        core_weight=core_weight,
        ignore_indice=ignore_indice,
        create_sigma_files=create_sigma_files
    )

    return obs


def generate_guess_atmos(
    rps_atmos_filepath,
    pixel_indices,
    rps,
    rps_name,
    write_path,
    previous_output_filename=None,
    smooth_thermo=None,
    include_b=False,
    smooth_b=None
):

    temp, vlos, vturb = get_rp_atmos(
        rps_atmos_filepath
    )

    ltau_scale = get_ltau_scale()

    m = sp.model(nx=pixel_indices.shape[1], ny=1, nt=1, ndep=ltau_scale.shape[0])

    m.ltau[:, :, :] = ltau_scale

    m.pgas[:, :, :] = 1.0

    if previous_output_filename is not None:

        fpo = h5py.File(previous_output_filename, 'r')

        temp = fpo['temp'][()]

        vlos = fpo['vlos'][()]

        vturb = fpo['vturb'][()]

        if smooth_thermo:

            temp = scipy.ndimage.gaussian_filter(temp, sigma=smooth_thermo / 2.355, axes=(1, 2))

            vlos = scipy.ndimage.gaussian_filter(vlos, sigma=smooth_thermo / 2.355, axes=(1, 2))

            vturb = scipy.ndimage.gaussian_filter(vturb, sigma=smooth_thermo / 2.355, axes=(1, 2))

        if pixel_indices.shape[0] == 4:

            m.temp[0, 0] = temp[pixel_indices[0], pixel_indices[1], pixel_indices[2]]

            m.vlos[0, 0] = vlos[pixel_indices[0], pixel_indices[1], pixel_indices[2]]

            m.vturb[0, 0] = vturb[pixel_indices[0], pixel_indices[1], pixel_indices[2]]

        else:

            m.temp[0, 0] = temp[0][pixel_indices[0], pixel_indices[1]]

            m.vlos[0, 0] = vlos[0][pixel_indices[0], pixel_indices[1]]

            m.vturb[0, 0] = vturb[0][pixel_indices[0], pixel_indices[1]]

    else:

        m.temp[0, 0] = temp[pixel_indices[-1]]

        m.vlos[0, 0] = vlos[pixel_indices[-1]]

        m.vturb[0, 0] = vturb[pixel_indices[-1]]

    write_filename = write_path / 'CA_SI_rps_{}_total_{}_initial_atmos.nc'.format(
        rps_name, pixel_indices.shape[1]
    )

    m.write(str(write_filename))


def write_profiles(
    pixel_indices,
    actual_filepath_ca,
    actual_filepath_si,
    rps,
    rps_name,
    write_path
):
    si_core_indice = [400, 656]

    ca2_core_indice = [0, 226]

    si_ignore_indice = [656, 872]

    # si_ignore_indice = [0, 872]

    wave_name_SI = 'SiI_10827'

    wave_name_CA = 'CaII_8542'

    wave_CA = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823

    wave_SI = np.arange(872, dtype=float) * 0.0144423 + 10818.6544101

    if pixel_indices.shape[1] == 0:
        print("None for RPS: {}".format('_'.join(map(str, rps))))
        sys.exit(0)

    obs_ca = make_observation_object_caller(
        actual_filepath=actual_filepath_ca,
        pixel_indices=pixel_indices,
        write_path=None,
        wave_name=wave_name_CA, wave=wave_CA,
        core_indice=ca2_core_indice,
        r_sigma=None,
        cont=4.227725e-05,
        factor=4,
        continuum_correction=1,
        all_weight=0.004,
        core_weight=0.0005,
        create_sigma_files=False
    )

    obs_si = make_observation_object_caller(
        actual_filepath=actual_filepath_si,
        pixel_indices=pixel_indices,
        write_path=None,
        wave_name=wave_name_SI, wave=wave_SI,
        core_indice=si_core_indice,
        r_sigma=None,
        cont=4.0709165e-05,
        factor=4,
        continuum_correction=1,
        all_weight=0.004,
        core_weight=0.002,
        ignore_indice=si_ignore_indice,
        create_sigma_files=False
    )

    all_profiles = obs_ca + obs_si

    writefilename = 'CA_SI_rps_{}_total_{}.nc'.format(
        rps_name, pixel_indices.shape[1]
    )

    all_profiles.write(
        write_path / writefilename
    )


def generate_actual_inversion_files_kmeans(
    actual_filepath_ca,
    actual_filepath_si,
    rps_atmos_filepath,
    rps,
    rps_name,
    label_keyname='final_labels_1',
    previous_output_filename=None,
    smooth_thermo=None,
    include_b=False,
    smooth_b=None
):

    write_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

    pixel_indices = save_pixel_indices_for_rps(
        write_path,
        rps,
        rps_name,
        label_key_name=label_keyname
    )

    write_profiles(
        pixel_indices=pixel_indices,
        actual_filepath_ca=actual_filepath_ca,
        actual_filepath_si=actual_filepath_si,
        rps=rps,
        rps_name=rps_name,
        write_path=write_path
    )
    
    generate_guess_atmos(
        rps_atmos_filepath=rps_atmos_filepath,
        pixel_indices=pixel_indices,
        rps=rps,
        rps_name=rps_name,
        write_path=write_path,
        previous_output_filename=previous_output_filename,
        smooth_thermo=smooth_thermo,
        include_b=include_b,
        smooth_b=smooth_b
    )


def merge_atmospheres(
    output_file,
    pixel_files,
    atmos_files,
    label_key_name='final_labels_1'
):

    keys = [
        'temp',
        'vlos',
        'vturb',
        'blong',
        'nne',
        'z',
        'ltau500',
        'pgas',
        'rho',
    ]

    with h5py.File(kmeans_file, 'r') as fl:
        labels = fl[label_key_name][()]

    f = h5py.File(output_file, 'w')
    outs = dict()

    ltau_scale = get_ltau_scale()

    for key in keys:
        if labels.ndim == 3:
            outs[key] = np.zeros((*labels.shape, ltau_scale.shape[0]), dtype=np.float64)
        else:
            outs[key] = np.zeros((1, *labels.shape, ltau_scale.shape[0]), dtype=np.float64)

    for pixel_file, atmos_file in zip(pixel_files, atmos_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(atmos_file, 'r')
        for key in keys:
            if pf['pixel_indices'].shape[0] == 4:
                a, b, c, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2], pf['pixel_indices'][3]
                outs[key][a, b, c] = af[key][0, 0]
            else:
                a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
                outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]
    f.close()


def merge_output_profiles(
    output_file,
    pixel_files,
    profile_files,
    label_key_name='final_labels_1'
):

    keys = [
        'profiles'
    ]

    with h5py.File(kmeans_file, 'r') as fl:
        labels = fl[label_key_name][()]

    with h5py.File(profile_files[0], 'r') as fp:
        wav = fp['wav'][()]
        nw = wav.shape[0]

    f = h5py.File(output_file, 'w')
    outs = dict()
    for key in keys:
        if labels.ndim == 3:
            outs[key] = np.zeros((*labels.shape, nw, 4), dtype=np.float64)
        else:
            outs[key] = np.zeros((1, *labels.shape, nw, 4), dtype=np.float64)

    for pixel_file, profile_file in zip(pixel_files, profile_files):
        pf = h5py.File(pixel_file, 'r')
        af = h5py.File(profile_file, 'r')
        for key in keys:
            if pf['pixel_indices'].shape[0] == 4:
                a, b, c, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2], pf['pixel_indices'][3]
                outs[key][a, b, c] = af[key][0, 0]
            else:
                a, b, rp = pf['pixel_indices'][0], pf['pixel_indices'][1], pf['pixel_indices'][2]
                outs[key][0, a, b] = af[key][0, 0]
        pf.close()
        af.close()

    for key in keys:
        f[key] = outs[key]

    f['wav'] = wav

    f.close()


if __name__ == '__main__':
    # make_rps()
    # make_rps_plots()
    # make_si_rps()
    # make_rps_plots(name='RPs_SI', cw=10827)
    # make_stic_inversion_files_si_ca_rps(rps=[98])
    # generate_input_atmos_file_from_falc(
    #     length=1,
    #     temp=[[-8, -5.5, -5, -4.5, -4, -3.5, -2, -1, 0, 2], [7000, 6500, 5500, 4500, 5500, 4500, 5000, 5500, 6500, 7000]],
    #     vlos=[[-8, -5.5, -5, -4.5, -4, -3.5, -2, -1, 0, 2], [20e5, 20e5, 20e5, 2e5, -10e5, -2e5, 2e5, 2e5, 2e5, 2e5]]
    #     )
    # generate_input_atmos_file_from_falc(length=1, in_file='/mnt/d/GRIS/KMeans-Inversions/ca_rps_stic_profiles_x_5_11_16_y_1.nc_level_5_alt_alt_cycle_1_t_7_vl_7_vt_4_falc_atmos.nc')
    # make_rps_inversion_plots()
    # make_rps_inversion_plots(rps=[98])  #, indexes=[91])
    # file='/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions/ca_rps_stic_profiles_x_5_11_16_y_1.nc_level_5_alt_alt_cycle_1_t_7_vl_7_vt_4_falc_atmos.nc'
    # file='/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions/ca_si_rps_stic_profiles_x_100_y_1_t_6_vlos_5_vturb_3_output_atmos.nc'
    # file='/mn/stornext/u3/harshm/Documents/Data/GRIS/KMeans-Inversions/Plots/Plots_54/ca_si_rps_stic_profiles_x_54_y_1_t_7_vlos_7_vturb_4_output_atmos.nc'
    # generate_input_atmos_file(
    #     length=1,
    #     temp=None,
    #     vlos=None,
    #     name='emission',
    #     file=file,
    #     index=0,
    #     vlos_multiplier=1,
    #     use_doppler=False,
    #     doppler_ltau=None,
    #     rps=None
    # )
    # calculate_continuum_correction()

    # combine_rps_atmos()

    actual_filepath_ca = base_path / 'spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    actual_filepath_si = base_path / 'spectralveil_corrected_25Apr25ARM1-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    rps_atmos_filepath = base_path / 'KMeans-Inversions' / 'combined_rps_atmos.nc'

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([1, 11, 12, 16, 17, 22, 25, 27, 28, 29, 31, 32, 34, 36, 41, 42, 48, 49, 51, 52, 54, 56, 57, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 77, 80, 85, 86, 87, 88, 89, 90, 92, 97, 98]),
    #     rps_name='ssf',
    #     label_keyname='final_labels_1'
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 30, 33, 35, 38, 39, 40, 43, 44, 45, 46, 47, 50, 53, 55, 58, 59, 66, 70, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84, 89, 91, 93, 94, 95, 96, 99]),
    #     rps_name='sft',
    #     label_keyname='final_labels_1'
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([6, 37]),
    #     rps_name='nsf',
    #     label_keyname='final_labels_1'
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([0]),
    #     rps_name='fsf',
    #     label_keyname='final_labels_1'
    # )

    data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

    pixel_files = [
        data_path / 'pixel_indices_ssf_total_78993.h5',
        data_path / 'pixel_indices_sft_total_35579.h5',
        data_path / 'pixel_indices_nsf_total_2647.h5',
        data_path / 'pixel_indices_fsf_total_261.h5'
    ]

    # atmos_files = [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_atmos.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_atmos.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_atmos.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_atmos.nc'
    # ]

    # profile_files =  [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_profs.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_profs.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_profs.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_profs.nc'
    # ]

    # output_merged_atmos = data_path / 'combined_output_atmos_no_B_cycle_1.nc'

    # output_merged_profs = data_path / 'combined_output_profs_no_B_cycle_1.nc'

    # merge_atmospheres(
    #     output_file=output_merged_atmos,
    #     pixel_files=pixel_files,
    #     atmos_files=atmos_files
    # )

    # merge_output_profiles(
    #     output_file=output_merged_profs,
    #     pixel_files=pixel_files,
    #     profile_files=profile_files
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([1, 11, 12, 16, 17, 22, 25, 27, 28, 29, 31, 32, 34, 36, 41, 42, 48, 49, 51, 52, 54, 56, 57, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 77, 80, 85, 86, 87, 88, 89, 90, 92, 97, 98]),
    #     rps_name='ssf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 30, 33, 35, 38, 39, 40, 43, 44, 45, 46, 47, 50, 53, 55, 58, 59, 66, 70, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84, 89, 91, 93, 94, 95, 96, 99]),
    #     rps_name='sft',
    #     label_keyname='final_labels_1',
        # previous_output_filename=output_merged_atmos,
        # smooth_thermo=3,
        # include_b=False,
        # smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([6, 37]),
    #     rps_name='nsf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([0]),
    #     rps_name='fsf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # atmos_files = [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_atmos_cycle_2.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_atmos_cycle_2.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_atmos_cycle_2.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_atmos_cycle_2.nc'
    # ]

    # profile_files =  [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_profs_cycle_2.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_profs_cycle_2.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_profs_cycle_2.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_profs_cycle_2.nc'
    # ]

    # output_merged_atmos = data_path / 'combined_output_atmos_no_B_cycle_2.nc'

    # output_merged_profs = data_path / 'combined_output_profs_no_B_cycle_2.nc'

    # merge_atmospheres(
    #     output_file=output_merged_atmos,
    #     pixel_files=pixel_files,
    #     atmos_files=atmos_files
    # )

    # merge_output_profiles(
    #     output_file=output_merged_profs,
    #     pixel_files=pixel_files,
    #     profile_files=profile_files
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([1, 11, 12, 16, 17, 22, 25, 27, 28, 29, 31, 32, 34, 36, 41, 42, 48, 49, 51, 52, 54, 56, 57, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 77, 80, 85, 86, 87, 88, 89, 90, 92, 97, 98]),
    #     rps_name='ssf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 30, 33, 35, 38, 39, 40, 43, 44, 45, 46, 47, 50, 53, 55, 58, 59, 66, 70, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84, 89, 91, 93, 94, 95, 96, 99]),
    #     rps_name='sft',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([6, 37]),
    #     rps_name='nsf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # generate_actual_inversion_files_kmeans(
    #     actual_filepath_ca=actual_filepath_ca,
    #     actual_filepath_si=actual_filepath_si,
    #     rps_atmos_filepath=rps_atmos_filepath,
    #     rps=np.array([0]),
    #     rps_name='fsf',
    #     label_keyname='final_labels_1',
    #     previous_output_filename=output_merged_atmos,
    #     smooth_thermo=3,
    #     include_b=False,
    #     smooth_b=None
    # )

    # atmos_files = [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_atmos_cycle_3.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_atmos_cycle_3.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_atmos_cycle_3.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_atmos_cycle_3.nc'
    # ]

    # profile_files =  [
    #     data_path / 'CA_SI_rps_ssf_total_78993_t_7_vlos_7_vturb_4_output_profs_cycle_3.nc',
    #     data_path / 'CA_SI_rps_sft_total_35579_t_6_vlos_5_vturb_3_output_profs_cycle_3.nc',
    #     data_path / 'CA_SI_rps_nsf_total_2647_t_9_vlos_7_vturb_4_output_profs_cycle_3.nc',
    #     data_path / 'CA_SI_rps_fsf_total_261_t_5_vlos_7_vturb_4_output_profs_cycle_3.nc'
    # ]

    output_merged_atmos = data_path / 'combined_output_atmos_no_B_cycle_3.nc'

    output_merged_profs = data_path / 'combined_output_profs_no_B_cycle_3.nc'

    # merge_atmospheres(
    #     output_file=output_merged_atmos,
    #     pixel_files=pixel_files,
    #     atmos_files=atmos_files
    # )

    # merge_output_profiles(
    #     output_file=output_merged_profs,
    #     pixel_files=pixel_files,
    #     profile_files=profile_files
    # )

    generate_actual_inversion_files_kmeans(
        actual_filepath_ca=actual_filepath_ca,
        actual_filepath_si=actual_filepath_si,
        rps_atmos_filepath=rps_atmos_filepath,
        rps=np.array([1, 11, 12, 16, 17, 22, 25, 27, 28, 29, 31, 32, 34, 36, 41, 42, 48, 49, 51, 52, 54, 56, 57, 60, 61, 62, 63, 64, 65, 67, 68, 69, 71, 72, 77, 80, 85, 86, 87, 88, 89, 90, 92, 97, 98]),
        rps_name='ssf',
        label_keyname='final_labels_1',
        previous_output_filename=output_merged_atmos,
        smooth_thermo=3,
        include_b=False,
        smooth_b=None
    )

    generate_actual_inversion_files_kmeans(
        actual_filepath_ca=actual_filepath_ca,
        actual_filepath_si=actual_filepath_si,
        rps_atmos_filepath=rps_atmos_filepath,
        rps=np.array([2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 18, 19, 20, 21, 23, 24, 26, 30, 33, 35, 38, 39, 40, 43, 44, 45, 46, 47, 50, 53, 55, 58, 59, 66, 70, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84, 89, 91, 93, 94, 95, 96, 99]),
        rps_name='sft',
        label_keyname='final_labels_1',
        previous_output_filename=output_merged_atmos,
        smooth_thermo=3,
        include_b=False,
        smooth_b=None
    )

    generate_actual_inversion_files_kmeans(
        actual_filepath_ca=actual_filepath_ca,
        actual_filepath_si=actual_filepath_si,
        rps_atmos_filepath=rps_atmos_filepath,
        rps=np.array([6, 37]),
        rps_name='nsf',
        label_keyname='final_labels_1',
        previous_output_filename=output_merged_atmos,
        smooth_thermo=3,
        include_b=False,
        smooth_b=None
    )

    generate_actual_inversion_files_kmeans(
        actual_filepath_ca=actual_filepath_ca,
        actual_filepath_si=actual_filepath_si,
        rps_atmos_filepath=rps_atmos_filepath,
        rps=np.array([0]),
        rps_name='fsf',
        label_keyname='final_labels_1',
        previous_output_filename=output_merged_atmos,
        smooth_thermo=3,
        include_b=False,
        smooth_b=None
    )

