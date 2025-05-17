import sys
sys.path.insert(1, '/mnt/d/Workrepo/stic/example/')
import numpy as np
import h5py
import sunpy.io
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import matplotlib.gridspec as gridspec
from prepare_data import *



base_path = Path(
    '/mnt/f/GRIS/'
)

base_path = Path('/mnt/f/GRIS')
kmeans_output_dir = base_path / 'K-Means'
input_files = [
    base_path / '25Apr25ARM2-003.fits',
    base_path / '25Apr25ARM2-004.fits'
]

kmeans_file = base_path / 'chosen_out_SV_100.h5'

rps_plot_write_dir = base_path / 'RPs_Plots'


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


def get_data(get_data=True, get_labels=True, get_rps=True, crop_indice=None):
    whole_data, labels, rps = None, None, None

    wave = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823

    if get_data:
        framerows = None

        for input_file in input_files:
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
        rps = f['rps'][()]

        # rps /= cont[0]
        rps[:, :, 1:4] /= rps[:, :, 0][:, :, np.newaxis]

    f.close()

    return whole_data, labels, rps, wave


def make_rps_plots(name='RPs'):
    whole_data, labels, rps, wave = get_data(crop_indice=None)

    k = 0

    color = 'black'

    cm = 'Blues'

    wave_x = np.arange(wave.size)

    xticks = list()

    xticks.append(np.argmin(np.abs(wave - 4226.73)))

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

                        in_bins_8542 = np.linspace(min_8542, max_8542, 1000)

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
            rps_plot_write_dir / 'RPs_{}.png'.format(k),
            format='png',
            dpi=300
        )

        plt.close('all')

        plt.clf()

        plt.cla()



def make_paper_rps_plots(name='RPs'):
    whole_data, labels, rps, wave = get_data(crop_indice=None)

    whole_data[:, :, 1:4] *= 100

    color = 'black'

    cm = 'Blues'

    wave_x = wave

    xticks = [4226.73]

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





if __name__ == '__main__':
    # make_rps()
    # make_halpha_rps()
    # plot_rp_map_fov()
    make_rps_plots()

    # make_paper_rps_plots()
