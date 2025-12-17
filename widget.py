import numpy as np
import h5py
import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from astropy.io import fits


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')

data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

# -----------------------------
# helpers
# -----------------------------
def wave_ca_8542(nw):
    return np.arange(nw) * 0.0109907 + 8540.67304823

def nearest(arr, val):
    return int(np.argmin(np.abs(arr - val)))

def build_mapping(w_obs, w_fit):
    idx = np.searchsorted(w_fit, w_obs)
    idx = np.clip(idx, 0, len(w_fit) - 1)
    idx_l = np.clip(idx - 1, 0, len(w_fit) - 1)
    use_l = np.abs(w_fit[idx_l] - w_obs) <= np.abs(w_fit[idx] - w_obs)
    return np.where(use_l, idx_l, idx)

# -----------------------------
# main tool
# -----------------------------
class SpectroTool:

    def __init__(self, obs_fits, atmos_h5, fitprof_h5):

        # ---- load data
        self.obs = fits.getdata(obs_fits)   # (t,s,y,x,w)
        self.temp = h5py.File(atmos_h5, "r")["temp"]
        self.vlos = h5py.File(atmos_h5, "r")["vlos"]
        self.vturb = h5py.File(atmos_h5, "r")["vturb"]
        self.blong = h5py.File(atmos_h5, "r")["blong"]
        self.ltau = h5py.File(atmos_h5, "r")["ltau500"][0,0,0,:]

        self.fit = h5py.File(fitprof_h5, "r")
        self.profiles = self.fit["profiles"]
        self.wav_fit = self.fit["wav"][:]

        self.nt, _, self.ny, self.nx, self.nw = self.obs.shape
        self.wave_obs = wave_ca_8542(self.nw)
        self.map_obs2fit = build_mapping(self.wave_obs, self.wav_fit)

        # ---- state
        self.t = 0
        self.y = 0
        self.x = 0
        self.wi = nearest(self.wave_obs, 8542.09)
        self.taui = nearest(self.ltau, -1.0)

        # ---- figures
        self.build_main_window()
        self.build_pixel_window()
        self.update_all()

        plt.show()

    # -------------------------
    # MAIN WINDOW
    # -------------------------
    def build_main_window(self):
        self.fig, self.ax = plt.subplots(3, 2, figsize=(12, 14))
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

        self.im_I = self.ax[0,0].imshow(self.obs[0,0,:,:,self.wi], origin="lower", cmap="gray")
        self.im_V = self.ax[0,1].imshow(self.obs[0,3,:,:,self.wi], origin="lower", cmap="RdBu_r")

        self.im_T = self.ax[1,0].imshow(self.temp[0,:,:,self.taui], origin="lower", cmap="viridis")
        self.im_U = self.ax[1,1].imshow(self.vlos[0,:,:,self.taui], origin="lower", cmap="RdBu_r")

        self.im_W = self.ax[2,0].imshow(self.vturb[0,:,:,self.taui], origin="lower", cmap="magma")
        self.im_B = self.ax[2,1].imshow(self.blong[0,:,:,self.taui], origin="lower", cmap="RdBu_r")

        for a in self.ax.ravel():
            a.set_xticks([]); a.set_yticks([])

        # sliders
        ax_t = plt.axes([0.15, 0.02, 0.65, 0.02])
        ax_w = plt.axes([0.15, 0.05, 0.65, 0.02])
        ax_tau = plt.axes([0.15, 0.08, 0.65, 0.02])

        self.s_t = Slider(ax_t, "time", 0, self.nt-1, valinit=self.t, valstep=1)
        self.s_w = Slider(ax_w, "λ idx", 0, self.nw-1, valinit=self.wi, valstep=1)
        self.s_tau = Slider(ax_tau, "logτ idx", 0, len(self.ltau)-1, valinit=self.taui, valstep=1)

        self.s_t.on_changed(self.on_slider)
        self.s_w.on_changed(self.on_slider)
        self.s_tau.on_changed(self.on_slider)

    # -------------------------
    # PIXEL WINDOW
    # -------------------------
    def build_pixel_window(self):
        self.figp, self.axp = plt.subplots(3, 2, figsize=(12, 12))

        (self.lIobs,) = self.axp[0,0].plot([], [], label="obs")
        (self.lIfit,) = self.axp[0,0].plot([], [], label="fit")
        self.axp[0,0].legend()

        (self.lVobs,) = self.axp[0,1].plot([], [], label="obs")
        (self.lVfit,) = self.axp[0,1].plot([], [], label="fit")
        self.axp[0,1].legend()

        (self.lT,) = self.axp[1,0].plot([], [])
        (self.lU,) = self.axp[1,1].plot([], [])
        (self.lW,) = self.axp[2,0].plot([], [])
        (self.lB,) = self.axp[2,1].plot([], [])

        for a in self.axp.ravel():
            a.grid(True)

    # -------------------------
    # callbacks
    # -------------------------
    def on_slider(self, val):
        self.t = int(self.s_t.val)
        self.wi = int(self.s_w.val)
        self.taui = int(self.s_tau.val)
        self.update_all()

    def onclick(self, event):
        if event.inaxes not in self.ax.ravel(): return
        if event.xdata is None or event.ydata is None: return

        self.x = int(event.xdata)
        self.y = int(event.ydata)
        self.update_pixel()

    # -------------------------
    # updates
    # -------------------------
    def update_main(self):
        self.im_I.set_data(self.obs[self.t,0,:,:,self.wi])
        self.im_V.set_data(self.obs[self.t,3,:,:,self.wi])
        self.im_T.set_data(self.temp[self.t,:,:,self.taui])
        self.im_U.set_data(self.vlos[self.t,:,:,self.taui])
        self.im_W.set_data(self.vturb[self.t,:,:,self.taui])
        self.im_B.set_data(self.blong[self.t,:,:,self.taui])

        lam = self.wave_obs[self.wi]
        tau = self.ltau[self.taui]

        self.ax[0,0].set_title(f"I @ {lam:.3f} Å")
        self.ax[0,1].set_title(f"V @ {lam:.3f} Å")
        self.ax[1,0].set_title(f"T @ logτ={tau:.2f}")
        self.ax[1,1].set_title(f"vlos @ logτ={tau:.2f}")
        self.ax[2,0].set_title(f"vturb @ logτ={tau:.2f}")
        self.ax[2,1].set_title(f"blong @ logτ={tau:.2f}")

        self.fig.canvas.draw_idle()

    def update_pixel(self):
        Iobs = self.obs[self.t,0,self.y,self.x,:]
        Vobs = self.obs[self.t,3,self.y,self.x,:]

        Ifit = self.profiles[self.t,self.y,self.x,:,0]
        Vfit = self.profiles[self.t,self.y,self.x,:,3]

        self.lIobs.set_data(self.wave_obs, Iobs)
        self.lIfit.set_data(self.wav_fit, Ifit)
        self.axp[0,0].relim(); self.axp[0,0].autoscale()

        self.lVobs.set_data(self.wave_obs, Vobs)
        self.lVfit.set_data(self.wav_fit, Vfit)
        self.axp[0,1].relim(); self.axp[0,1].autoscale()

        self.lT.set_data(self.ltau, self.temp[self.t,self.y,self.x,:])
        self.lU.set_data(self.ltau, self.vlos[self.t,self.y,self.x,:])
        self.lW.set_data(self.ltau, self.vturb[self.t,self.y,self.x,:])
        self.lB.set_data(self.ltau, self.blong[self.t,self.y,self.x,:])

        for a in self.axp[1:, :].ravel():
            a.relim(); a.autoscale()

        self.figp.canvas.draw_idle()

    def update_all(self):
        self.update_main()
        self.update_pixel()


if __name__ == '__main__':

    actual_filepath_ca = base_path / 'spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    output_merged_atmos = data_path / 'combined_output_atmos_cycle_B_3.nc'

    output_merged_profs = data_path / 'combined_output_profs_cycle_B_3.nc'

    SpectroTool(
        obs_fits=actual_filepath_ca,
        atmos_h5=output_merged_atmos,
        fitprof_h5=output_merged_profs,
    )
