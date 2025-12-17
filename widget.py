import numpy as np
import h5py
import matplotlib.pyplot as plt
from astropy.io import fits
import ipywidgets as widgets
from IPython.display import display
from pathlib import Path


base_path = Path('/mn/stornext/u3/harshm/Documents/Data/GRIS')
data_path = base_path / 'KMeans-Inversions' / 'fulldata_inversions'

# ---------------------------
# helpers
# ---------------------------
def wave_ca_8542(nwave: int = 1000) -> np.ndarray:
    return np.arange(nwave, dtype=float) * 0.0109907 + 8540.67304823

def nearest_index_1d(arr: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr - value)))

def build_nearest_mapping(from_wave: np.ndarray, to_wave: np.ndarray) -> np.ndarray:
    """
    For each element in from_wave, returns index in to_wave that is nearest.
    Uses searchsorted for speed.
    """
    from_wave = np.asarray(from_wave)
    to_wave = np.asarray(to_wave)

    # ensure sorted to_wave
    if np.any(np.diff(to_wave) < 0):
        raise ValueError("fitted 'wav' must be sorted ascending for this mapping method.")

    idx = np.searchsorted(to_wave, from_wave, side="left")
    idx = np.clip(idx, 0, len(to_wave) - 1)

    idx_left = np.clip(idx - 1, 0, len(to_wave) - 1)
    choose_left = np.abs(to_wave[idx_left] - from_wave) <= np.abs(to_wave[idx] - from_wave)
    out = np.where(choose_left, idx_left, idx)
    return out.astype(np.int64)

def finite_minmax(a: np.ndarray):
    m = np.isfinite(a)
    if not np.any(m):
        return 0.0, 1.0
    return float(np.min(a[m])), float(np.max(a[m]))

# ---------------------------
# main tool
# ---------------------------
class SpectroAtmosViewer:
    """
    Interactive viewer:
      - Observed spectral cube: FITS (t, stokes, y, x, w_obs) with w_obs = wave_CA
      - Atmosphere: HDF5 keys temp/vlos/vturb/blong (t, y, x, ntau), ltau500 (t, y, x, ntau) but constant grid
      - Fitted profiles: HDF5 key profiles (t, y, x, w_fit, stokes), wav (w_fit,)
    """
    def __init__(
        self,
        obs_fits_path: str,
        atmos_h5_path: str,
        fitprof_h5_path: str,
        stokes_I: int = 0,
        stokes_V: int = 3,
        initial_tyx=(0, 0, 0),
        cmap_I="gray",
        cmap_V="RdBu_r",
        cmap_temp="viridis",
        cmap_vlos="RdBu_r",
        cmap_vturb="magma",
        cmap_blong="RdBu_r",
    ):
        self.obs_fits_path = obs_fits_path
        self.atmos_h5_path = atmos_h5_path
        self.fitprof_h5_path = fitprof_h5_path
        self.stokes_I = stokes_I
        self.stokes_V = stokes_V

        self.cmaps = dict(
            I=cmap_I, V=cmap_V,
            temp=cmap_temp, vlos=cmap_vlos, vturb=cmap_vturb, blong=cmap_blong
        )

        # Open files (keep open for speed)
        self.hdul = fits.open(self.obs_fits_path, memmap=True)
        self.obs = self.hdul[0].data  # (t, s, y, x, w)
        if self.obs.ndim != 5:
            raise ValueError(f"Observed FITS must be 5D (t,s,y,x,w). Got shape {self.obs.shape}.")

        self.atmos = h5py.File(self.atmos_h5_path, "r")
        self.fit = h5py.File(self.fitprof_h5_path, "r")

        # Datasets
        self.temp = self.atmos["temp"]
        self.vlos = self.atmos["vlos"]
        self.vturb = self.atmos["vturb"]
        self.blong = self.atmos["blong"]
        self.ltau500_4d = self.atmos["ltau500"]  # (t,y,x,ntau) but constant grid

        self.profiles = self.fit["profiles"]     # (t,y,x,w_fit,stokes)
        self.wav_fit = np.array(self.fit["wav"][:], dtype=float)  # (w_fit,)

        # Wavelength grids
        self.wave_obs = wave_ca_8542(self.obs.shape[-1])

        # Mapping: wave_obs -> wav_fit (nearest indices)
        self.map_obs_to_fit = build_nearest_mapping(self.wave_obs, self.wav_fit)

        # Shapes
        self.nt = min(self.obs.shape[0], self.temp.shape[0], self.profiles.shape[0])
        self.ny = self.obs.shape[2]
        self.nx = self.obs.shape[3]
        self.nw_obs = self.obs.shape[-1]
        self.ntau = self.temp.shape[-1]

        # logtau grid (constant, use [0,0,0,:] as you said)
        self.ltau = np.array(self.ltau500_4d[0, 0, 0, :], dtype=float)

        # initial selection
        t0, y0, x0 = initial_tyx
        self.sel_t = int(np.clip(t0, 0, self.nt - 1))
        self.sel_y = int(np.clip(y0, 0, self.ny - 1))
        self.sel_x = int(np.clip(x0, 0, self.nx - 1))

        # default slider positions
        self.w_idx = nearest_index_1d(self.wave_obs, 8542.09)  # start near line core
        self.tau_idx = nearest_index_1d(self.ltau, -1.0)

        # Precompute fixed color limits (fast-ish but still scans all times)
        # If this is too heavy, you can switch to percentiles on a subset of frames.
        self._compute_fixed_limits()

        # Build UI + figures
        self._build_ui()
        self._build_figures()
        self._connect()
        self._redraw_all()

    def close(self):
        try: self.hdul.close()
        except: pass
        try: self.atmos.close()
        except: pass
        try: self.fit.close()
        except: pass

    # ---------------------------
    # fixed color limits
    # ---------------------------
    def _compute_fixed_limits(self):
        # Map panels: I-map, V-map at chosen wavelength vary with wavelength.
        # For stability while sliding wavelength, we compute limits per Stokes using *all wavelengths and times*.
        # Atmos maps per parameter use all times and all tau indices? We’ll fix per parameter across all times+taus.
        # This ensures no flicker even when logtau slider changes.

        # Observed I/V (scan all t and w, keep min/max)
        I_min = I_max = None
        V_min = V_max = None
        for t in range(self.nt):
            Iframe = self.obs[t, self.stokes_I, :, :, :]
            Vframe = self.obs[t, self.stokes_V, :, :, :]
            a, b = finite_minmax(Iframe)
            c, d = finite_minmax(Vframe)
            I_min = a if I_min is None else min(I_min, a)
            I_max = b if I_max is None else max(I_max, b)
            V_min = c if V_min is None else min(V_min, c)
            V_max = d if V_max is None else max(V_max, d)

        # Atmos (scan all t and tau)
        def scan_atmos(dset):
            mn = mx = None
            for t in range(self.nt):
                frame = dset[t, :, :, :]  # (y,x,tau)
                a, b = finite_minmax(frame)
                mn = a if mn is None else min(mn, a)
                mx = b if mx is None else max(mx, b)
            return mn, mx

        temp_min, temp_max = scan_atmos(self.temp)
        vlos_min, vlos_max = scan_atmos(self.vlos)
        vturb_min, vturb_max = scan_atmos(self.vturb)
        blong_min, blong_max = scan_atmos(self.blong)

        self.lims = dict(
            I=(I_min, I_max),
            V=(V_min, V_max),
            temp=(temp_min, temp_max),
            vlos=(vlos_min, vlos_max),
            vturb=(vturb_min, vturb_max),
            blong=(blong_min, blong_max),
        )

    # ---------------------------
    # UI
    # ---------------------------
    def _build_ui(self):
        self.slider_t = widgets.IntSlider(
            value=self.sel_t, min=0, max=self.nt - 1, step=1,
            description="time", continuous_update=True, readout=True
        )
        self.slider_w = widgets.IntSlider(
            value=self.w_idx, min=0, max=self.nw_obs - 1, step=1,
            description="λ idx", continuous_update=True, readout=True
        )
        self.slider_tau = widgets.IntSlider(
            value=self.tau_idx, min=0, max=self.ntau - 1, step=1,
            description="logτ idx", continuous_update=True, readout=True
        )

        self.lbl_wave = widgets.Label("")
        self.lbl_tau = widgets.Label("")
        self.lbl_sel = widgets.Label("")

        controls = widgets.VBox([
            self.slider_t,
            widgets.HBox([self.slider_w, self.lbl_wave]),
            widgets.HBox([self.slider_tau, self.lbl_tau]),
            self.lbl_sel
        ])
        display(controls)

        # Observe slider changes
        self.slider_t.observe(self._on_slider_change, names="value")
        self.slider_w.observe(self._on_slider_change, names="value")
        self.slider_tau.observe(self._on_slider_change, names="value")

    # ---------------------------
    # figures
    # ---------------------------
    def _build_figures(self):
        # Main window: 3x2 maps
        self.fig_main, self.ax_main = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
        self.fig_main.canvas.header_visible = False  # in some notebook frontends

        # Inspector window: 3x2 plots
        self.fig_pix, self.ax_pix = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)

        # Initialize artists for main maps
        # Row 1: I, V
        self.im_I = self.ax_main[0, 0].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                             cmap=self.cmaps["I"], vmin=self.lims["I"][0], vmax=self.lims["I"][1])
        self.im_V = self.ax_main[0, 1].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                             cmap=self.cmaps["V"], vmin=self.lims["V"][0], vmax=self.lims["V"][1])

        # Row 2-3: temp, vlos, vturb, blong at selected logtau
        self.im_temp = self.ax_main[1, 0].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                                 cmap=self.cmaps["temp"], vmin=self.lims["temp"][0], vmax=self.lims["temp"][1])
        self.im_vlos = self.ax_main[1, 1].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                                 cmap=self.cmaps["vlos"], vmin=self.lims["vlos"][0], vmax=self.lims["vlos"][1])
        self.im_vturb = self.ax_main[2, 0].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                                  cmap=self.cmaps["vturb"], vmin=self.lims["vturb"][0], vmax=self.lims["vturb"][1])
        self.im_blong = self.ax_main[2, 1].imshow(np.zeros((self.ny, self.nx)), origin="lower",
                                                  cmap=self.cmaps["blong"], vmin=self.lims["blong"][0], vmax=self.lims["blong"][1])

        for a in self.ax_main.ravel():
            a.set_xticks([]); a.set_yticks([])

        # Add 6 colorbars (one per map)
        self.cb_I = self.fig_main.colorbar(self.im_I, ax=self.ax_main[0, 0], fraction=0.046, pad=0.02)
        self.cb_V = self.fig_main.colorbar(self.im_V, ax=self.ax_main[0, 1], fraction=0.046, pad=0.02)
        self.cb_temp = self.fig_main.colorbar(self.im_temp, ax=self.ax_main[1, 0], fraction=0.046, pad=0.02)
        self.cb_vlos = self.fig_main.colorbar(self.im_vlos, ax=self.ax_main[1, 1], fraction=0.046, pad=0.02)
        self.cb_vturb = self.fig_main.colorbar(self.im_vturb, ax=self.ax_main[2, 0], fraction=0.046, pad=0.02)
        self.cb_blong = self.fig_main.colorbar(self.im_blong, ax=self.ax_main[2, 1], fraction=0.046, pad=0.02)

        # Pixel selection markers (crosshair)
        self.cross = []
        for ax in self.ax_main.ravel():
            ln1 = ax.axhline(self.sel_y, lw=0.8)
            ln2 = ax.axvline(self.sel_x, lw=0.8)
            self.cross.append((ln1, ln2))

        # Inspector artists: spectra overlays and strat plots
        # spectra (row1)
        self.ax_pix[0, 0].set_title("Spectra overlay: Stokes I")
        self.ax_pix[0, 1].set_title("Spectra overlay: Stokes V")
        (self.line_I_obs,) = self.ax_pix[0, 0].plot([], [], lw=1.2, label="obs (wave_CA)")
        (self.line_I_fit,) = self.ax_pix[0, 0].plot([], [], lw=1.2, label="fit (wav)")
        self.ax_pix[0, 0].legend(loc="best")

        (self.line_V_obs,) = self.ax_pix[0, 1].plot([], [], lw=1.2, label="obs (wave_CA)")
        (self.line_V_fit,) = self.ax_pix[0, 1].plot([], [], lw=1.2, label="fit (wav)")
        self.ax_pix[0, 1].legend(loc="best")

        # stratification (rows2-3)
        titles = [["temp vs logτ", "vlos vs logτ"], ["vturb vs logτ", "blong vs logτ"]]
        self.strat_axes = [self.ax_pix[1, 0], self.ax_pix[1, 1], self.ax_pix[2, 0], self.ax_pix[2, 1]]
        for ax, ttl in zip(self.strat_axes, [titles[0][0], titles[0][1], titles[1][0], titles[1][1]]):
            ax.set_title(ttl)
            ax.set_xlabel("logτ")
            ax.grid(True, alpha=0.3)

        (self.line_temp,) = self.ax_pix[1, 0].plot([], [], lw=1.5)
        (self.line_vlos,) = self.ax_pix[1, 1].plot([], [], lw=1.5)
        (self.line_vturb,) = self.ax_pix[2, 0].plot([], [], lw=1.5)
        (self.line_blong,) = self.ax_pix[2, 1].plot([], [], lw=1.5)

        display(self.fig_main)
        display(self.fig_pix)

    # ---------------------------
    # interactions
    # ---------------------------
    def _connect(self):
        self.cid = self.fig_main.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event):
        if event.inaxes is None:
            return
        # Only accept clicks in main axes
        if event.inaxes not in self.ax_main.ravel().tolist():
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))
        if (0 <= x < self.nx) and (0 <= y < self.ny):
            self.sel_x = x
            self.sel_y = y
            self._update_crosshairs()
            self._redraw_pixel_inspector()

    def _update_crosshairs(self):
        for (h, v) in self.cross:
            h.set_ydata([self.sel_y, self.sel_y])
            v.set_xdata([self.sel_x, self.sel_x])

    def _on_slider_change(self, change):
        self.sel_t = int(self.slider_t.value)
        self.w_idx = int(self.slider_w.value)
        self.tau_idx = int(self.slider_tau.value)
        self._redraw_all()

    # ---------------------------
    # data pulls
    # ---------------------------
    def _get_obs_maps(self, t: int, w_idx: int):
        I = self.obs[t, self.stokes_I, :, :, w_idx]
        V = self.obs[t, self.stokes_V, :, :, w_idx]
        return I, V

    def _get_atmos_maps(self, t: int, tau_idx: int):
        T = self.temp[t, :, :, tau_idx]
        U = self.vlos[t, :, :, tau_idx]
        W = self.vturb[t, :, :, tau_idx]
        B = self.blong[t, :, :, tau_idx]
        return T, U, W, B

    def _get_pixel_spectra(self, t: int, y: int, x: int):
        # observed spectra on wave_CA
        I_obs = self.obs[t, self.stokes_I, y, x, :]
        V_obs = self.obs[t, self.stokes_V, y, x, :]

        # fitted spectra on wav_fit
        I_fit = self.profiles[t, y, x, :, self.stokes_I]
        V_fit = self.profiles[t, y, x, :, self.stokes_V]

        # also sample fit onto obs grid using nearest mapping (useful for “direct comparison”)
        I_fit_on_obs = I_fit[self.map_obs_to_fit]
        V_fit_on_obs = V_fit[self.map_obs_to_fit]

        return I_obs, V_obs, I_fit, V_fit, I_fit_on_obs, V_fit_on_obs

    def _get_pixel_strat(self, t: int, y: int, x: int):
        return (
            self.temp[t, y, x, :],
            self.vlos[t, y, x, :],
            self.vturb[t, y, x, :],
            self.blong[t, y, x, :],
        )

    # ---------------------------
    # redraws
    # ---------------------------
    def _update_labels(self):
        lam = self.wave_obs[self.w_idx]
        tau = self.ltau[self.tau_idx]
        self.lbl_wave.value = f"λ = {lam:.4f} Å"
        self.lbl_tau.value = f"logτ = {tau:.3f}"
        self.lbl_sel.value = f"Selected (t,y,x)=({self.sel_t},{self.sel_y},{self.sel_x})"

        self.ax_main[0, 0].set_title(f"Stokes I @ {lam:.4f} Å")
        self.ax_main[0, 1].set_title(f"Stokes V @ {lam:.4f} Å")
        self.ax_main[1, 0].set_title(f"temp @ logτ={tau:.3f}")
        self.ax_main[1, 1].set_title(f"vlos @ logτ={tau:.3f}")
        self.ax_main[2, 0].set_title(f"vturb @ logτ={tau:.3f}")
        self.ax_main[2, 1].set_title(f"blong @ logτ={tau:.3f}")

    def _redraw_main(self):
        I, V = self._get_obs_maps(self.sel_t, self.w_idx)
        T, U, W, B = self._get_atmos_maps(self.sel_t, self.tau_idx)

        self.im_I.set_data(I)
        self.im_V.set_data(V)
        self.im_temp.set_data(T)
        self.im_vlos.set_data(U)
        self.im_vturb.set_data(W)
        self.im_blong.set_data(B)

        self.fig_main.canvas.draw_idle()

    def _redraw_pixel_inspector(self):
        t, y, x = self.sel_t, self.sel_y, self.sel_x

        I_obs, V_obs, I_fit, V_fit, I_fit_on_obs, V_fit_on_obs = self._get_pixel_spectra(t, y, x)
        temp1d, vlos1d, vturb1d, blong1d = self._get_pixel_strat(t, y, x)

        # Spectra overlays:
        # Plot obs on wave_obs; fit on wav_fit
        self.line_I_obs.set_data(self.wave_obs, I_obs)
        self.line_I_fit.set_data(self.wav_fit, I_fit)
        self.ax_pix[0, 0].relim(); self.ax_pix[0, 0].autoscale_view()

        self.line_V_obs.set_data(self.wave_obs, V_obs)
        self.line_V_fit.set_data(self.wav_fit, V_fit)
        self.ax_pix[0, 1].relim(); self.ax_pix[0, 1].autoscale_view()

        self.ax_pix[0, 0].set_title(f"Stokes I (t,y,x)=({t},{y},{x})")
        self.ax_pix[0, 1].set_title(f"Stokes V (t,y,x)=({t},{y},{x})")
        self.ax_pix[0, 0].set_xlabel("Wavelength (Å)")
        self.ax_pix[0, 1].set_xlabel("Wavelength (Å)")

        # Strat plots
        self.line_temp.set_data(self.ltau, temp1d)
        self.line_vlos.set_data(self.ltau, vlos1d)
        self.line_vturb.set_data(self.ltau, vturb1d)
        self.line_blong.set_data(self.ltau, blong1d)

        for ax in self.strat_axes:
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(self.ltau.min(), self.ltau.max())

        self.fig_pix.canvas.draw_idle()

    def _redraw_all(self):
        self._update_labels()
        self._update_crosshairs()
        self._redraw_main()
        self._redraw_pixel_inspector()


if __name__ == '__main__':

    actual_filepath_ca = base_path / 'spectralveil_corrected_25Apr25ARM2-004.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits'

    output_merged_atmos = data_path / 'combined_output_atmos_cycle_B_3.nc'

    output_merged_profs = data_path / 'combined_output_profs_cycle_B_3.nc'

    viewer = SpectroAtmosViewer(
        obs_fits_path=actual_filepath_ca,
        atmos_h5_path=output_merged_atmos,
        fitprof_h5_path=output_merged_profs,
        initial_tyx=(0, 0, 0),
    )
