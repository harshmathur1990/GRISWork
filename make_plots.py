import numpy as np
import sunpy.io
import matplotlib.pyplot as plt
import matplotlib

def make_plots_for_dkist_proposal():

    data, header = sunpy.io.read_file('/mnt/f/GRIS/25Apr25ARM2-003.fits')[0]

    trdata = data[:, :, ::-1, ::-1, :]

    wave = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823

    pix_scale_x = 0.12913

    pix_scale_y = 0.1875

    nx, ny = trdata.shape[3], trdata.shape[2]

    extent = [0, nx * pix_scale_x, 0, ny * pix_scale_y]

    points = [
        (13, 31)
    ]

    plt.close('all')

    plt.clf()

    plt.cla()

    font = {'size': 8}

    matplotlib.rc('font', **font)

    fig, axs = plt.subplots(2, 2, figsize=(7, 5), layout='constrained')

    axs[0][0].imshow(trdata[0, 0, :, :, 900], cmap='gray', origin='lower', extent=extent)

    axs[1][0].imshow(trdata[0, 0, :, :, 129], cmap='gray', origin='lower', extent=extent)

    axs[0][1].plot(wave, trdata[0, 0, points[0][0], points[0][1], :])

    axs[1][1].plot(wave, trdata[0, 3, points[0][0], points[0][1], :] / trdata[0, 0, points[0][0], points[0][1], :])

    axs[1][1].set_ylim(-0.13, 0.13)

    axs[0][1].set_ylabel(r'Stokes $I$')

    axs[1][1].set_ylabel(r'Stokes $V/I$')

    axs[1][1].set_xlabel(r'Wavelength [$\mathrm{\AA}$]')

    axs[1][0].set_xlabel(r'$x$ [arcsec]')

    axs[1][0].set_ylabel(r'$y$ [arcsec]')

    for point in points:
        axs[0][0].scatter(point[1] * pix_scale_x, point[0] * pix_scale_y, marker='x', color='r')

        axs[1][0].scatter(point[1] * pix_scale_x, point[0] * pix_scale_y, marker='x', color='r')

    plt.savefig('figures/dkist_proposal.pdf', format='pdf')

    plt.savefig('figures/dkist_proposal.png', format='png', dpi=600)


if __name__ == '__main__':
    make_plots_for_dkist_proposal()
