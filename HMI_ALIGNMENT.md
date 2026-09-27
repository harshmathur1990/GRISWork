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
frames matched to one HMI file cannot overwrite each other. The existing
`align_sdo_from_hmi_continuum.py` still consumes cropped continuum files and
uses nominal filename clocks; it has not been migrated to this manifest and
should not be treated as using the new exact UTC matching.

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
