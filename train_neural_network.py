import h5py
import tensorflow as tf
import scipy.ndimage
import numpy as np
from pathlib import Path

tf.config.threading.set_intra_op_parallelism_threads(24)  # ops like matmul
tf.config.threading.set_inter_op_parallelism_threads(24)  # parallel ops


# ---- Simulated disk-backed data (replace with file-based access) ----
n_samples = 504*504  # Example: too large for memory
n_wavelengths = 1000


base_path = Path('/mnt/f/BifrostRun')
gris_base = Path('/mnt/f/GRIS')

hdf5_path = base_path / 'BIFROST_en024048_hion_0_504_0_504_supplementary_outputs_multi3d_pops_pb_rates_ca_2.nc'
temp_hdf5 = base_path / 'BIFROST_en024048_hion_0_504_0_504_ltau_temp.nc'
vlos_hdf5 = base_path / 'BIFROST_en024048_hion_0_504_0_504_ltau_velocity_z.nc'


filter_file = gris_base / 'spectral_veil_estimated_profile_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits.h5'
falc_mu_1_file = gris_base / 'FALC_GRIS_IFU_1.nc'
falc_mu_0p79_file = gris_base / 'FALC_GRIS_IFU_0p79.nc'


wavegrid1 = np.arange(1000, dtype=float) * 0.0109907 + 8540.67304823

node_positions = [-6, -5, -4, -3, -2, -1, 0]


def increase_sampling(wave, factor=2, extra_points=8):
    # Infer original step (assumes uniform spacing)
    step = wave[1] - wave[0]
    
    # New step = half of original
    new_step = step / factor
    
    # Shift min and max outward by extra_points in *new* step
    new_min = wave[0] - extra_points * new_step
    new_max = wave[-1] + extra_points * new_step
    
    # Generate the new high-resolution wavelength array
    wave_highres = np.arange(new_min, new_max + new_step/2, new_step)

    return wave_highres


# Simulate disk-backed data with on-demand generator
class HDF5InterpolationGenerator:
    def __init__(self, batch_size=32, shuffle=True):
        self.f = h5py.File(hdf5_path, 'r')
        self.temp_f = h5py.File(temp_hdf5, 'r')
        self.vlos_f = h5py.File(vlos_hdf5, 'r')
        self.filter_f = h5py.File(filter_file, 'r')
        self.falc_mu_1_f = h5py.File(falc_mu_1_file, 'r')
        self.falc_mu_0p79_f = h5py.File(falc_mu_0p79_file, 'r')

        falc_at_wavegrid_1 = np.interp(wavegrid1, self.falc_mu_1_f['wav'][()], self.falc_mu_1_f['profiles'][0, 0, 0, :, 0])

        falc_at_wavegrid_0p79 = np.interp(wavegrid1, self.falc_mu_0p79_f['wav'][()], self.falc_mu_0p79_f['profiles'][0, 0, 0, :, 0])

        ctl_variation = falc_at_wavegrid_0p79 / falc_at_wavegrid_1

        self.node_locations = list()

        for node in node_positions:
            self.node_locations.append(
                np.argmin(
                    np.abs(
                        self.temp_f['ltau500'][()] - node
                    )
                )
            )

        self.node_locations = np.array(self.node_locations)

        self.profiles = self.f['profiles_CA']  # Do NOT slice yet!
        self.wavegrid2 = self.f['wave_CA'][()]
        
        self.wavegrid1 = wavegrid1
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Determine dimensions from file (without loading all)
        self.ny = self.profiles.shape[1]
        self.nx = self.profiles.shape[2]
        self.indices = [(i, j) for i in range(self.ny) for j in range(self.nx)]

        # Define overlapping wavelength range
        wl_min = max(self.wavegrid1.min(), self.wavegrid2.min())
        wl_max = min(self.wavegrid1.max(), self.wavegrid2.max())
        self.wavegrid_common = self.wavegrid1[(self.wavegrid1 >= wl_min) & (self.wavegrid1 <= wl_max)]

        self.ctl_variation = np.interp(self.wavegrid_common, wavegrid1, ctl_variation)

        self.highres_factor = 2
        self.wave_highres = increase_sampling(self.wavegrid_common, factor=self.highres_factor)

        ind_synth = np.argmin(
            np.abs(
                self.falc_mu_0p79_f['wav'][()] - self.wavegrid1[-1]
            )
        )

        self.cont_factor = self.falc_mu_0p79_f['profiles'][0, 0, 0, ind_synth, 0] * 4.227725e-08

    def get_labels(self, i, j):
        temp = self.temp_f['temperature'][i, j][self.node_locations] / 1000
        vlos = self.vlos_f['velocity_z'][i, j][self.node_locations] / 1000
        labels = np.zeros(temp.shape[0] + vlos.shape[0], dtype=np.float64)
        labels[0:temp.shape[0]] = temp
        labels[temp.shape[0]:] = vlos

        return labels

    def __call__(self):
        while True:
            if self.shuffle:
                np.random.shuffle(self.indices)

            for start in range(0, len(self.indices), self.batch_size):
                batch_indices = self.indices[start:start + self.batch_size]
                batch_spectra = []
                batch_labels = []

                for i, j in batch_indices:
                    # On-the-fly read from HDF5 file: shape (800,)
                    spectrum = self.profiles[0, i, j, :, 0]  # Just Stokes I

                    # Interpolate
                    spectrum_interp = np.interp(self.wave_highres, self.wavegrid2, spectrum)
                    spectrum_interp = scipy.ndimage.gaussian_filter1d(
                        spectrum_interp,
                        self.filter_f['sigma_in_pixels'][()] * self.highres_factor
                    )

                    spectrum_interp = np.interp(self.wavegrid_common, self.wave_highres, spectrum_interp)

                    spectrum_interp = spectrum_interp * self.ctl_variation

                    spectrum_interp = spectrum_interp / self.cont_factor

                    batch_spectra.append(spectrum_interp)
                    batch_labels.append(self.get_labels(i, j))

                yield np.stack(batch_spectra), np.stack(batch_labels)

data_generator = HDF5InterpolationGenerator(batch_size=32, shuffle=True)

# ---- Create tf.data.Dataset ----
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(32, 492,), dtype=tf.float32),
        tf.TensorSpec(shape=(32, 14,), dtype=tf.float32)
    )
)

# ---- Shuffle, batch, and split ----
# Shuffle and cache a buffer subset (not full data)
dataset = dataset.shuffle(buffer_size=1000)

# Split into train, val, test (e.g., 80/10/10)
train_size = int(0.8 * n_samples)
val_size = int(0.1 * n_samples)
test_size = n_samples - train_size - val_size

train_ds = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.skip(train_size).take(val_size).prefetch(tf.data.AUTOTUNE)
test_ds = dataset.skip(train_size + val_size).prefetch(tf.data.AUTOTUNE)

# ---- Define and Compile Model ----
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(492,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(14)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ---- Train ----
model.fit(train_ds, validation_data=val_ds, epochs=50)

# ---- Evaluate ----
loss, mae = model.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}")
