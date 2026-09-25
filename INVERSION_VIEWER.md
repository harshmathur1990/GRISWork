# GRIS inversion comparison viewer

Install in your scientific Python environment:

```sh
python -m pip install -r requirements-viewer.txt
python inversion_viewer.py
```

The opening dialog selects the observed Ca FITS cube, timestamp CSV, optional
merged atmosphere/profile files and optional aligned SDO root (containing
`HMI/Continuum`, `HMI/Magnetogram`, `AIA/171`, etc.). Any subdirectory with
recognizable SDO FITS filenames becomes a selectable channel. Files are read only.

You can also supply paths directly:

```sh
python inversion_viewer.py \
  --observation /path/to/spectralveil_corrected_25Apr25ARM2-003.fits_squarred_pixels.fits_aligned_downsampled_streamed.fits \
  --atmosphere /path/to/combined_output_atmos_cycle_B_3.nc \
  --profiles /path/to/combined_output_profs_cycle_B_3.nc \
  --aligned-root /path/to/aligned_SDO \
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
Inversion grids and time dimensions must match the observation. All maps use
lower-origin pixel coordinates with no additional transpose, flip or reprojection.
SDO images with different spatial dimensions are reported as unavailable.

By default, SDO files are matched to the nearest telescope time using **nominal
filename clocks**, preserving `align_sdo_from_hmi_continuum.py` conventions,
including its treatment of HMI TAI filenames. An optional UTC mode converts HMI
filename TAI clocks to UTC; AIA filename clocks are already UTC. Matching is by
source image time; `ALNREF` identifies the spatial alignment reference and is
shown separately. The default maximum absolute offset is 60 seconds and is
editable. Each SDO panel shows the selected filename, clock, signed offset and
alignment reference. Missing/out-of-range/invalid frames clear the panel rather
than retaining a stale image. A source image can be reused at adjacent telescope
frames if it is the closest image within tolerance.

## Embedding and tests

`ComparisonWidget(store, rows=2, columns=2)` is a PySide6 `QWidget` and can be
embedded in another Qt application. Construct `DataStore` from
`inversion_viewer_data`; the caller owns it and must call `store.close()` after
closing the widget. The command-line application handles cleanup automatically.

```sh
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

Tests use small generated FITS/HDF5 data to check array orientation, units,
timestamp validation, SDO clock matching and tolerances, panel controls, grid
changes, playback and stale-image clearing. Real-data validation still requires
the external telescope, inversion and aligned SDO files.
