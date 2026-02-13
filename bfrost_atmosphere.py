import sys
sys.path.insert(1, '/mn/stornext/u3/harshm/Documents/WorkRepo/stic/example')

import numpy as np
import h5py
from prepare_data import *
from pathlib import Path
from tqdm import tqdm

data_path = Path('/mn/stornext/d9/data/harshm/halpha_output')

# ------------------------------------------------------------
# LTau scale
# ------------------------------------------------------------
def get_ltau_scale():
    taumin = -7.8
    taumax = 1.0
    dtau  = 0.14
    ntau  = int((taumax - taumin) / dtau) + 1
    return np.arange(ntau, dtype='float64')/(ntau-1.0) * (taumax-taumin) + taumin

ltau_new = get_ltau_scale()
ndep_new = ltau_new.size

# ------------------------------------------------------------
# Open files (lazy loading via h5py)
# ------------------------------------------------------------
f  = h5py.File(data_path / 'BIFROST_en024048_hion_0_504_0_504_supplementary_outputs_multi3d_pops_pb_rates.nc', 'r')
fa = h5py.File(data_path / 'BIFROST_en024048_hion_0_504_0_504.nc', 'r')

ltau_ds = f['ltau500']
temp_ds = fa['temperature']
vz_ds   = fa['velocity_z']
bz_ds   = fa['B_z']

nx = ltau_ds.shape[1]
ny = ltau_ds.shape[2]
ndep_old = 467

# ------------------------------------------------------------
# Create STiC model
# ------------------------------------------------------------
m = sp.model(nx=nx, ny=ny, nt=1, ndep=ndep_new)
m.ltau[:, :, :] = ltau_new
m.pgas[:, :, :] = 1.0
m.vturb[:, :, :] = 0.0   # constant

# ------------------------------------------------------------
# Chunking parameters
# ------------------------------------------------------------
chunk_size = 32   # tune (16–64 safe on Saga)
# memory per chunk ~ chunk_size * ny * depth * fields

for i0 in tqdm(range(0, nx, chunk_size), desc="Processing columns"):
    
    i1 = min(i0 + chunk_size, nx)

    # ---- Read only chunk ----
    ltau_old = ltau_ds[0, i0:i1, :, :]
    temp_old = temp_ds[0, i0:i1, :, :ndep_old]
    vz_old   = vz_ds[0, i0:i1, :, :ndep_old]
    bz_old   = bz_ds[0, i0:i1, :, :ndep_old]

    # ---- Unit conversions ----
    vz_old *= 1e2    # m/s -> cm/s
    bz_old *= 1e4      # Tesla -> Gauss

    nxc = i1 - i0
    ncol = nxc * ny

    # ---- Flatten ----
    ltau_old = ltau_old.reshape(ncol, ndep_old)
    temp_old = temp_old.reshape(ncol, ndep_old)
    vz_old   = vz_old.reshape(ncol, ndep_old)
    bz_old   = bz_old.reshape(ncol, ndep_old)

    # ---- Sort (safe guard) ----
    sort_idx = np.argsort(ltau_old, axis=1)
    ltau_old = np.take_along_axis(ltau_old, sort_idx, axis=1)
    temp_old = np.take_along_axis(temp_old, sort_idx, axis=1)
    vz_old   = np.take_along_axis(vz_old, sort_idx, axis=1)
    bz_old   = np.take_along_axis(bz_old, sort_idx, axis=1)

    # ---- Interpolation ----
    temp_new = np.empty((ncol, ndep_new))
    vz_new   = np.empty((ncol, ndep_new))
    bz_new   = np.empty((ncol, ndep_new))

    for k in range(ndep_new):

        tau_val = ltau_new[k]

        idx = np.sum(ltau_old < tau_val, axis=1)
        idx = np.clip(idx, 1, ndep_old - 1)

        x0 = ltau_old[np.arange(ncol), idx - 1]
        x1 = ltau_old[np.arange(ncol), idx]

        w = (tau_val - x0) / (x1 - x0)

        for field_old, field_new in zip(
            (temp_old, vz_old, bz_old),
            (temp_new, vz_new, bz_new)
        ):
            y0 = field_old[np.arange(ncol), idx - 1]
            y1 = field_old[np.arange(ncol), idx]
            field_new[:, k] = y0 + w * (y1 - y0)

    # ---- Reshape back ----
    temp_new = temp_new.reshape(nxc, ny, ndep_new)
    vz_new   = vz_new.reshape(nxc, ny, ndep_new)
    bz_new   = bz_new.reshape(nxc, ny, ndep_new)

    # ---- Write into model ----
    m.temp[0, i0:i1, :, :] = temp_new
    m.vlos[0, i0:i1, :, :] = vz_new
    m.Bln[0, i0:i1, :, :]  = bz_new

# ------------------------------------------------------------
# Write output
# ------------------------------------------------------------
m.write(data_path / 'bifrost.nc')

f.close()
fa.close()
