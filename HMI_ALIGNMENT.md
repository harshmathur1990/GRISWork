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
- `gris_wcs/gris_0000.fits`, etc.: the selected native GRIS wavelength image with
  its fitted solar WCS. This is the reusable alignment, independent of a crop.

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

root = Path('/mnt/f/GRIS/aligned_SDO/HMI/Continuum')
record = json.loads((root / 'alignment.json').read_text())['frames'][0]
hmi = sunpy.map.Map(root / record['registered_hmi'])
gris = sunpy.map.Map(root / record['gris_map'])

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
