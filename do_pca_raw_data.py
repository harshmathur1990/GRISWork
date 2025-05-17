import sys

import matplotlib.pyplot as plt

sys.path.insert(1, '/mnt/d/Workrepo/stic/example/')
from sklearn.decomposition import PCA
import h5py
import sunpy.io
import numpy as np
from pathlib import Path
from prepare_data import *


base_path = Path('/mnt/f/GRIS')
kmeans_output_dir = base_path / 'K-Means'
input_files = [
    base_path / '25Apr25ARM2-003.fits',
    base_path / '25Apr25ARM2-004.fits'
]


def do_pca():
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

    n_components = {
        0: 10,
        1: 128,
        2: 176,
        3: 123
    }

    stokes_V = framerows[:, 3]

    mu = np.mean(stokes_V, axis=0)

    pca = PCA(n_components=stokes_V.shape[1])

    nComps = 0
    nCompe = n_components[3]
    Xhat = np.dot(pca.fit_transform(stokes_V)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_V_1 = Xhat[0:t1 * y1 * x1].reshape((t1, y1, x1, Xhat.shape[1]))

    new_stokes_V_2 = Xhat[t1 * y1 * x1:].reshape((y2, x2, Xhat.shape[1]))

    stokes_Q = framerows[:, 1]

    mu = np.mean(stokes_Q, axis=0)

    pca = PCA(n_components=stokes_Q.shape[1])

    nComps = 0
    nCompe = n_components[1]
    Xhat = np.dot(pca.fit_transform(stokes_Q)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_Q_1 = Xhat[0:t1 * y1 * x1].reshape((t1, y1, x1, Xhat.shape[1]))

    new_stokes_Q_2 = Xhat[t1 * y1 * x1:].reshape((y2, x2, Xhat.shape[1]))

    stokes_U = framerows[:, 2]

    mu = np.mean(stokes_U, axis=0)

    pca = PCA(n_components=stokes_U.shape[1])

    nComps = 0
    nCompe = n_components[2]
    Xhat = np.dot(pca.fit_transform(stokes_U)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_U_1 = Xhat[0:t1 * y1 * x1].reshape((t1, y1, x1, Xhat.shape[1]))

    new_stokes_U_2 = Xhat[t1 * y1 * x1:].reshape((y2, x2, Xhat.shape[1]))

    stokes_I = framerows[:, 0]

    mu = np.mean(stokes_I, axis=0)

    pca = PCA(n_components=stokes_I.shape[1])

    nComps = 0
    nCompe = n_components[0]
    Xhat = np.dot(pca.fit_transform(stokes_I)[:, nComps:nCompe], pca.components_[nComps:nCompe, :])
    Xhat += mu

    new_stokes_I_1 = Xhat[0:t1 * y1 * x1].reshape((t1, y1, x1, Xhat.shape[1]))

    new_stokes_I_2 = Xhat[t1 * y1 * x1:].reshape((y2, x2, Xhat.shape[1]))

    new_data_1 = np.zeros(
        (
            t1,
            y1,
            x1,
            4,
            new_stokes_I_1.shape[3]
        ),
        dtype=np.float64
    )

    new_data_2 = np.zeros(
        (
            y2,
            x2,
            4,
            new_stokes_I_2.shape[2]
        ),
        dtype=np.float64
    )

    new_data_1[:, :, :, 0] = new_stokes_I_1
    new_data_1[:, :, :, 1] = new_stokes_Q_1
    new_data_1[:, :, :, 2] = new_stokes_U_1
    new_data_1[:, :, :, 3] = new_stokes_V_1

    new_data_2[:, :, 0] = new_stokes_I_2
    new_data_2[:, :, 1] = new_stokes_Q_2
    new_data_2[:, :, 2] = new_stokes_U_2
    new_data_2[:, :, 3] = new_stokes_V_2

    wfilename = '25Apr25ARM2-003.fits_pca.fits'
    wfilepath = base_path / wfilename
    sunpy.io.write_file(wfilepath, new_data_1, dict(), overwrite=True)

    wfilename = '25Apr25ARM2-004.fits_pca.fits'
    wfilepath = base_path / wfilename
    sunpy.io.write_file(wfilepath, new_data_2, dict(), overwrite=True)


if __name__ == '__main__':
    do_pca()