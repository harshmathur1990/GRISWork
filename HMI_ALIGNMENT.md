# Preserve the HMI / GRIS alignment

Run `alignment_GUI_HMI.py` in your existing SunPy/aiapy environment. Set the
input/output directories at the bottom of the script. `timestamps_path` points
to `serie_timestamps.csv` beside the script by default. There is no start-time
or cadence parameter: each zero-based GRIS frame maps to CSV `series_index =
frame_index + 1`. The CSV must have exactly one ordered UTC timestamp per frame.
If the cube is a subset or has been reordered, supply a correspondingly indexed
CSV. HMI `_TAI` filename times are converted to UTC before nearest-time matching.
The time difference is recorded; no temporal interpolation is performed.

Align each frame, then click **Save Data**. The output directory contains:

- `alignment.json`: source paths, exact GRIS times, matched HMI times, signed
  time differences, fitted centres, wavelength indices, and relative output paths.
- `registered/`: full images after `register()`, including their WCS, saved once
  per matched HMI input. Original HMI files are untouched.
- `gris_wcs/gris_0000.hdr`, etc.: text FITS headers containing the fitted solar
  WCS and native GRIS dimensions. No GRIS image data is saved again.

The WCS uses the matched HMI observer and coordinate reference time. `GRISDATE`
separately records the actual GRIS timestamp. Do not replace the WCS observation
time when overlaying on that HMI frame. The fit assumes the input GRIS cube has
already been oriented correctly and has square 0.135 arcsec pixels; it fits only
translation. It does not correct solar evolution or differential rotation
between observations. Other data must share the same GRIS pixel grid (including
orientation and shape) to reuse this WCS unchanged.

The GUI now samples HMI through WCS and preserves native GRIS pixel centres
when upsampling. Saved WCS and preview share the same geometry. Because the old
preview used endpoint interpolation and hard-coded HMI coordinates, redo manual
alignment rather than treating old fitted centres as an equivalent solution.

## Contours on a full registered HMI image

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sunpy.map
from astropy.io import fits

root = Path('/mnt/f/GRIS/aligned_SDO/HMI/Continuum')
manifest = json.loads((root / 'alignment.json').read_text())
record = manifest['frames'][0]
hmi = sunpy.map.Map(root / record['registered_hmi'])
header = fits.Header.fromtextfile(root / record['gris_header'])
with fits.open(manifest['source_gris'], memmap=True) as hdus:
    cube = hdus[0].data
    t, w = record['frame_index'], record['wavelength_index']
    data = (cube[t, 0, :, :, w] if cube.ndim == 5 else cube[0, :, :, w]).copy()
assert data.shape == (header['NAXIS2'], header['NAXIS1'])
gris = sunpy.map.Map(data, header)

fig = plt.figure()
ax = fig.add_subplot(projection=hmi)
hmi.plot(axes=ax, cmap='gray')
# Replace gris.data with any 2-D quantity on the same native GRIS grid.
# Choose physically appropriate contour levels for that quantity.
ax.contour(gris.data, levels=[0.8 * gris.data.max()],
           transform=ax.get_transform(gris.wcs), colors='red')
plt.show()
```

Here WCSAxes transforms GRIS pixel positions into HMI pixel positions, so no
HMI crop is needed. For a different registered HMI image, use its map as the
axes projection; differences in observation time may require an explicit solar
rotation treatment.

Set `save_crops=True` in the `animate()` call to additionally export HMI on the
GRIS grid. These optional filenames include the GRIS frame index, so two GRIS
frames matched to one HMI file cannot overwrite each other. `align_sdo_from_hmi_continuum.py` does not use these optional crops; it reads
the GRIS WCS headers directly and exports full registered SDO images instead.

## Align with two marked features

1. Select the time frame and a GRIS continuum wavelength where the same features
   are visible in HMI.
2. Click **Feature 1**, then click the centre of that feature in the bottom GRIS
   image and the top registered HMI image, in either order. Orange markers label
   the first pair. Flickering pauses so that the bottom image stays on GRIS.
3. Click **Feature 2** and mark a second distinct feature in both images. Cyan
   markers label this pair. Re-click either image to replace the selected mark.
4. Click **Auto Align**. It first fits a common translation to both pairs through
   WCS, then searches for the translation with the highest mean normalized
   correlation in patches around the two GRIS features. Scale and orientation
   remain fixed. The fitted centre and both correlation scores appear below the
   images, and the flicker preview resumes for visual inspection.
5. Click **Save Data** to persist the fitted WCS headers. The manifest also records
   the marker positions in native pixels, patch correlations, landmark fit, and
   correlation refinement for automatically aligned frames. No GRIS images are
   written again.

**Patch px** is the patch half-width in native GRIS pixels (default 20, giving a
41 × 41 patch away from edges). **Search ″** bounds the correlation refinement
in each solar coordinate around the landmark estimate (default ±1.5 arcsec).
Brightness offsets and contrast scaling are removed separately in each patch.
Masked/NaN pixels are excluded; each patch needs at least 25 valid samples and
80% HMI coverage. A correlation below 0.3 in either patch, or a best match at the
search boundary, leaves the existing alignment unchanged and displays a message.
These thresholds are basic checks, not an uncertainty estimate; inspect the
flicker result, particularly for repetitive or evolving features.

**Clear Marks** clears the current frame/wavelength's points. **Resume Flicker**
exits marking mode without applying an alignment. Marks are independent for each
frame and wavelength and remain available when returning to that image in the
same session. Manual centre changes invalidate the previous automatic-fit report.
Matplotlib toolbar pan/zoom takes precedence over feature marking; turn it off
before placing a mark. Outside marking mode, clicking HMI still sets the centre
manually as before.

## Automatically align the whole time sequence

Mark both feature pairs on any reference frame, choose the continuum wavelength,
and click **Align All Frames**. There is no need to visit or mark every frame.
The tool aligns the reference first, then processes later frames in order and
returns to the reference to process earlier frames in reverse order. It uses the
selected wavelength throughout the batch and independently matches HMI to each
frame's CSV timestamp.

The same native GRIS patch locations are reused: this assumes the input GRIS cube
is already spatially aligned, as in the default input filename. Each successful
centre seeds the next frame, with predicted HMI pixel positions recalculated
through that frame's registered WCS. Correlation then refines the translation
within **Search ″** of the seed. This follows gradual motion without reusing raw
HMI pixel coordinates. It does not independently track large feature motion
within an unaligned GRIS cube.

Progress appears below the images. During a batch, editing and saving are disabled,
and the batch button becomes **Stop Alignment**; stopping takes effect between
frames and retains completed fits. No files are written until **Save Data**.

A failed frame keeps its previous alignment and is flagged for review. Subsequent
frames start from the last successful fit. If the reference itself fails, the
batch stops. Failed or unprocessed frame indices appear in the status/console;
visiting those frames also shows the reason. Saving is blocked until these frames
have been aligned successfully or adjusted manually. Simply visiting a frame does
not accept its inherited alignment. You can re-mark a failed frame and use
**Auto Align**, or rerun **Align All Frames** from a better reference.

After completion, inspect the flicker preview using the time slider, then click
**Save Data**. The headers and manifest include each frame's independent fit,
correlation scores, reference index, and seed frame. GRIS image data is still not
saved again. A new batch replaces successful fits, including any earlier manual
fits, with results from the selected reference and wavelength.


## Register other SDO channels from the GRIS headers

`align_sdo_from_hmi_continuum.py` reads the `.hdr` files in
`<aligned-root>/HMI/Continuum/gris_wcs`. It does not require the GUI's JSON
manifest or any cropped continuum FITS files. To use a different header directory,
pass `--gris-wcs /path/to/gris_wcs`.

```bash
python align_sdo_from_hmi_continuum.py \
    --raw-root /mn/stornext/d9/data/harshm/GRISData/SDO \
    --aligned-root /mn/stornext/d9/data/harshm/GRISData/aligned_SDO
```

The default channels are HMI/Continuum, HMI/Magnetogram, AIA/171, AIA/1600, and
AIA/304. Use `--channels AIA/171 AIA/1600` to select a subset. Each GRIS frame is
matched to exactly one nearest source in each selected channel using the exact
`timestamp_utc` from `serie_timestamps.csv`. The CSV is read directly; pass
`--timestamps /path/to/serie_timestamps.csv` to choose another file. CSV
`series_index - 1` must match the header's `GRISIDX`, with one header per CSV row.
The headers supply the spatial WCS; CSV timestamps determine temporal matching. HMI filename TAI clocks are converted to UTC; AIA filenames retain their
UTC clock, including fractional seconds. By default the closest available
observation is selected regardless of its offset. Set `--max-time-delta SECONDS`
only if you want to reject more distant matches. Only observations selected by
at least one GRIS frame are registered; intermediate or extra SDO exposures are
not exported. Two GRIS frames may share one registered file while retaining
separate manifest entries and signed time offsets.

Outputs are:

- `<aligned-root>/<channel>/registered/<original-name>.fits`: the full image
  after `register()`, with its own WCS and observation time. A source shared by
  several GRIS frames is saved once. Existing registered files are reused unless
  `--overwrite` is given. `--no-register` is only for already-registered inputs.
- `<aligned-root>/alignment.json`: one combined manifest for the selected channels.
  Its `frames` entries contain `gris_header`, `timestamp_utc`, and a `channels`
  dictionary. Each channel entry includes the source path, `registered_sdo` path,
  UTC filename timestamp, signed SDO-minus-GRIS time offset, and processing status.
  Header and output paths are relative to the manifest's directory.

The root manifest is refreshed on every non-dry run for the selected channels.
It records the source CSV path and one entry per GRIS frame. Files left in
`registered/` by earlier runs are not deleted, but unselected ones are omitted
from the new manifest.
The GUI's separate `HMI/Continuum/alignment.json` remains untouched. Existing
crops are not deleted, but this script no longer writes cropped HMI/AIA files,
reprojects onto the GRIS grid, or changes an observation's time by differential
rotation. The old `--no-differential-rotation` flag has been removed. GRIS header
files and image data are not duplicated or modified.

Use `--dry-run` to inspect associations without loading image arrays or writing
files (the CSV and GRIS text headers are read). Missing/too-distant observations are marked
`unmatched`; processing errors are marked `failed`, with the reason in the
manifest. Such runs return exit code 1 while keeping successful results. The
manifest never links a failed registration as a usable output.

To overlay existing GRIS data on a selected channel:

```python
root = Path('/mn/stornext/d9/data/harshm/GRISData/aligned_SDO')
record = json.loads((root / 'alignment.json').read_text())['frames'][0]
channel = record['channels']['AIA/171']
assert channel['status'] in ('written', 'existing')
sdo = sunpy.map.Map(root / channel['registered_sdo'])
header = fits.Header.fromtextfile(root / record['gris_header'])
# gris_data_2d is your existing quantity for this frame on the native GRIS grid.
assert gris_data_2d.shape == tuple(record['gris_shape'])
gris = sunpy.map.Map(gris_data_2d, header)
fig = plt.figure()
ax = fig.add_subplot(projection=sdo)
sdo.plot(axes=ax)
ax.contour(gris.data, levels=[0.8 * gris.data.max()],
           transform=ax.get_transform(gris.wcs), colors='red')
plt.show()
```

This overlays coordinates without evolving features between observation times.
For analyses needing solar-rotation compensation, apply that explicitly during
plotting or a separate analysis step, using the saved observation times.
