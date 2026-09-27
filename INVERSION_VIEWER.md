# GRIS inversion comparison viewer

Install in your scientific Python environment:

```sh
python -m pip install -r requirements-viewer.txt
python inversion_viewer.py
```

The opening dialog selects the observed Ca FITS cube, timestamp CSV, optional
merged atmosphere/profile files and optional aligned SDO root (containing
`HMI/Continuum`, `HMI/Magnetogram`, `AIA/171`, etc.). The root `alignment.json`
selects the saved nearest-UTC match per GRIS frame/channel. Without a manifest,
only `<channel>/registered/*.fits` are indexed for nearest-UTC matching. Older
cropped products are ignored. Files are read only. GRIS headers are discovered
from the manifest or `HMI/Continuum/gris_wcs`; the dialog and `--gris-wcs` option
can select a different header directory.

You can also supply paths directly:

```sh
python inversion_viewer.py \
  --observation /path/to/spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits \
  --atmosphere /path/to/combined_output_atmos_cycle_B_3.nc \
  --profiles /path/to/combined_output_profs_cycle_B_3.nc \
  --aligned-root /path/to/aligned_SDO \
  --timestamps serie_timestamps.csv --rows 2 --columns 3
```

```
py inversion_viewer.py \
  --observation /mn/stornext/d9/data/harshm/GRISData/spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits \
  --atmosphere /mn/stornext/d9/data/harshm/fulldata_inversions/combined_output_atmos_cycle_B_3.nc \
  --profiles /mn/stornext/d9/data/harshm/fulldata_inversions/combined_output_profs_cycle_B_3.nc \
  --aligned-root /mn/stornext/d9/data/harshm/GRISData/aligned_SDO \
  --timestamps serie_timestamps.csv --rows 2 --columns 3
```
- Choose rows and columns, then **Apply grid**. Existing panels retain settings
  in row-major order; shrinking removes the trailing panels.
- Choose a timestamp or drag the shared slider. **Play/Pause**, previous/next,
  adjustable FPS and **Loop** control playback. FPS is a requested fixed display
  cadence, not the telescope's acquisition cadence; disk reads/rendering can slow it.
- Each panel independently selects a source. Open its **Settings** to adjust
  wavelength, depth, Stokes and colors; collapse settings to give images more room.
  Observation and fitted profiles offer
  Stokes I/Q/U/V, a zero-based wavelength index and wavelength in Å. Entering a
  wavelength selects the nearest sample and displays its actual wavelength.
- Atmosphere panels offer all matching 4-D fields and a zero-based depth index.
  The displayed log τ and optional nearest-value selection use `ltau500[0,0,0,:]`
  (or a 1-D `ltau500`). The selected index is used at every pixel, without depth
  interpolation. Temperature is shown in kK, velocities in km/s and `blong` in G;
  other quantities retain native units. Profiles also retain native intensity units.
- Each panel has a colormap, a colorbar, Matplotlib pan/zoom/save controls, and
  automatic 1st–99th percentile contrast, held limits, or manual min/max.
  **Hold limits** freezes the current limits across time; changing the source,
  wavelength, Stokes or atmosphere field resets them.

## Data and matching conventions

The observed cube is `(time, Stokes, y, x, wavelength)`, fitted `profiles` is
`(time, y, x, wavelength, Stokes)` with a separate 1-D `wav`, and atmosphere
fields are `(time, y, x, depth)`. Observation wavelengths default to
`8540.67304823 + index * 0.0109907 Å`, matching the existing inversion script;
`--wave-start` and `--wave-step` can override this calibration. Cubes are sliced
on demand using FITS memory mapping and HDF5 datasets, without loading entire
inversion cubes into memory.

CSV row 1 maps to cube time index 0; `series_index` must be consecutive and the
CSV must have exactly as many timestamps as the observation (30 for this run).
Inversion grids and time dimensions must match the observation and GRIS headers.
All spatial axes and cursor coordinates use arcseconds. With GRIS WCS headers,
observations, fitted profiles, and inversion maps use the per-frame solar WCS;
no transpose or flip is applied to these arrays. Without headers, standalone
GRIS/inversion panels explicitly show relative offsets from the field centre at
0.135 arcsec per sample. Absolute SDO overlays require the saved GRIS WCS.

Every SDO panel displays the **full registered HMI/AIA image**, with its native
WCS and arcsecond axes. There is no 50″ crop or resampling of the SDO background.
A cyan **rectangle marks the GRIS field of view**, transformed from that frame's
GRIS WCS into the SDO image coordinates. Intensity contours are not drawn.
The viewer does not apply differential solar rotation between observations.

For optional flickering, open **Settings** to select the **GRIS overlay wavelength
sample** or wavelength in Å; the default is the first observed wavelength sample.
Choose a continuum wavelength when checking alignment against HMI continuum.

Each SDO panel has a **Flicker GRIS** button. It switches every 500 ms between SDO
and the selected GRIS image inside the GRIS footprint; the larger surrounding
SDO field remains visible. GRIS is shown in grayscale with independent 1–99%
contrast. The colourbar continues to describe the SDO image. Click **Stop flicker**
to return to SDO. Switching away from SDO, removing a panel, closing the viewer,
or encountering an unavailable frame stops that panel's flicker timer. Changing
time updates its data and WCS without resetting the flicker phase.

The manifest's explicit associations take precedence over extra files left on
disk. Manifest timestamps must agree with the CSV, and header indices and shapes
must agree with the GRIS cube. In the fallback directory scan, HMI filename TAI
clocks are converted to UTC and AIA clocks are already UTC. The former nominal
clock mode has been removed. The maximum absolute offset is 60 seconds by default
and is editable. Each SDO panel shows its filename, source UTC and signed offset.
Missing/out-of-range/invalid frames clear both the image and overlays rather than
retaining a stale frame. One registered source may serve adjacent GRIS frames.

## Embedding and tests

`ComparisonWidget(store, rows=2, columns=2)` is a PySide6 `QWidget` and can be
embedded in another Qt application. Construct `DataStore` from
`inversion_viewer_data`; the caller owns it and must call `store.close()` after
closing the widget. The command-line application handles cleanup automatically.

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Tests use small generated FITS/HDF5 data to check array orientation, units,
timestamp validation, UTC/manifest matching and tolerances, preservation of full SDO arrays, rotated
WCS rectangle placement, flicker lifecycle, panel controls, grid changes, playback
and stale-image clearing. Real-data validation still requires
the external telescope, inversion and aligned SDO files.
