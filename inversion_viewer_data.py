"""Lazy readers and time matching for the GRIS comparison viewer."""
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
from functools import lru_cache

import h5py
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel
from scipy.ndimage import map_coordinates
import sunpy.map
from align_sdo_from_hmi_continuum import utc_time_from_name


def utc_datetime(value):
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def read_timestamps(path):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    indices = [int(row['series_index']) for row in rows]
    if indices != list(range(1, len(rows) + 1)) or not rows:
        raise ValueError('CSV series_index must run consecutively from 1 in cube order.')
    times = [utc_datetime(row['timestamp_utc']) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('CSV timestamps must be strictly increasing.')
    return times


@dataclass(frozen=True)
class SDOFrame:
    path: Path
    utc: datetime


@dataclass
class DisplayImage:
    data: np.ndarray
    unit: str
    detail: str
    wcs: WCS
    absolute: bool
    gris_data: np.ndarray | None = None
    gris_wcs: WCS | None = None
    overlay: np.ndarray | None = None


class DataStore:
    """Keep cubes on disk; only selected 2-D slices are materialized."""
    def __init__(self, observation, timestamps, atmosphere=None, profiles=None,
                 aligned_root=None, gris_wcs=None, wave_start=8540.67304823, wave_step=0.0109907):
        self.resources = ExitStack()
        self.warnings = []
        try:
            self.times = read_timestamps(timestamps)
            hdul = self.resources.enter_context(fits.open(observation, memmap=True))
            self.obs = hdul[0].data
            if self.obs.ndim != 5 or self.obs.shape[1] not in (1, 4):
                raise ValueError('Observation must have shape (time, Stokes, y, x, wavelength).')
            self.nt, self.ns, self.ny, self.nx, self.nw = self.obs.shape
            if self.nt != len(self.times):
                raise ValueError(f'Observation has {self.nt} frames, CSV has {len(self.times)} rows.')
            self.wave_obs = wave_start + np.arange(self.nw) * wave_step
            if not np.all(np.isfinite(self.wave_obs)) or wave_step == 0:
                raise ValueError('Observation wavelength calibration must be finite with nonzero step.')
            self.atmos = self.resources.enter_context(h5py.File(atmosphere, 'r')) if atmosphere else None
            self.fit = self.resources.enter_context(h5py.File(profiles, 'r')) if profiles else None
            self.parameters = []
            if self.atmos is not None:
                self.parameters = [k for k, v in self.atmos.items()
                                   if isinstance(v, h5py.Dataset) and v.ndim == 4
                                   and v.shape[:3] == (self.nt, self.ny, self.nx)]
                preferred = ['temp', 'vlos', 'vturb', 'blong']
                self.parameters.sort(key=lambda k: (preferred.index(k) if k in preferred else 4, k))
                if not self.parameters or 'ltau500' not in self.atmos:
                    raise ValueError('Atmosphere must contain matching (time,y,x,depth) fields and ltau500.')
                tau = self.atmos['ltau500']
                if tau.ndim not in (1, 4):
                    raise ValueError('ltau500 must be 1-D or (time,y,x,depth).')
                self.tau = np.asarray(tau[:] if tau.ndim == 1 else tau[0, 0, 0, :])
                if not self.tau.size or not np.all(np.isfinite(self.tau)):
                    raise ValueError('Reference optical depths must be nonempty and finite.')
                if any(self.atmos[k].shape[-1] != len(self.tau) for k in self.parameters):
                    raise ValueError('Atmosphere depth dimensions disagree.')
            if self.fit is not None:
                shape = self.fit['profiles'].shape
                self.wave_fit = np.asarray(self.fit['wav'][:])
                if (len(shape) != 5 or shape[:3] != (self.nt, self.ny, self.nx)
                        or shape[3] != self.wave_fit.size or shape[4] not in (1, 4)
                        or self.wave_fit.ndim != 1):
                    raise ValueError('Profiles must have shape (time,y,x,wavelength,Stokes) and 1-D wav.')
                if not self.wave_fit.size or not np.all(np.isfinite(self.wave_fit)):
                    raise ValueError('Fitted wavelengths must be nonempty and finite.')
            self.sdo = {}
            self.sdo_matches = None
            self.gris_headers = {}
            root = Path(aligned_root) if aligned_root else None
            if root is not None and not root.is_dir():
                raise ValueError(f'Aligned SDO directory does not exist: {root}')
            header_dir = Path(gris_wcs) if gris_wcs else (root / 'HMI/Continuum/gris_wcs' if root else None)
            if gris_wcs and not header_dir.is_dir():
                raise ValueError(f'GRIS WCS directory does not exist: {header_dir}')
            manifest_path = root / 'alignment.json' if root else None
            if manifest_path and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                self.sdo_matches = {}
                seen = set()
                for row in manifest['frames']:
                    index = int(row['frame_index'])
                    if index in seen or not 0 <= index < self.nt:
                        raise ValueError('Manifest has duplicate or out-of-range GRIS indices.')
                    seen.add(index)
                    if utc_datetime(row['timestamp_utc']) != self.times[index]:
                        raise ValueError(f'Manifest timestamp disagrees with CSV for frame {index}.')
                    if not gris_wcs:
                        self._add_header(root / row['gris_header'], index)
                    for channel, record in row['channels'].items():
                        self.sdo.setdefault(channel, [])
                        if record['status'] not in ('written', 'existing') or not record.get('registered_sdo'):
                            self.sdo_matches[channel, index] = record.get('reason', 'SDO match unavailable')
                            continue
                        frame = SDOFrame(root / record['registered_sdo'], utc_datetime(record['timestamp_utc']))
                        self.sdo_matches[channel, index] = frame
                        if frame not in self.sdo[channel]:
                            self.sdo[channel].append(frame)
                if len(seen) != self.nt:
                    raise ValueError('Manifest must contain every GRIS time frame.')
            if header_dir and header_dir.is_dir() and not self.gris_headers:
                for path in sorted(header_dir.glob('*.hdr')):
                    self._add_header(path)
            if self.gris_headers and len(self.gris_headers) != self.nt:
                raise ValueError('GRIS WCS headers must cover every observation frame.')
            if not self.gris_headers:
                self.warnings.append('No GRIS WCS supplied: GRIS/inversions use offsets from the field centre '
                                     'at 0.135 arcsec/sample; absolute SDO overlays are unavailable.')
            if root and self.sdo_matches is None:
                # Only full registered products; ignore old crops and unselected files
                # whenever the authoritative manifest is available.
                for path in sorted(root.glob('*/*/registered/*.fits')):
                    try:
                        stamp = utc_time_from_name(path).replace(tzinfo=None)
                        channel = path.parent.parent.relative_to(root).as_posix()
                        self.sdo.setdefault(channel, []).append(SDOFrame(path, stamp))
                    except ValueError as exc:
                        self.warnings.append(f'Skipped {path.name}: {exc}')
            if root and not self.sdo:
                self.warnings.append('No registered SDO images found.')
        except Exception:
            self.close()
            raise

    @property
    def sources(self):
        return (['Observation'] + (['Fitted profiles'] if self.fit is not None else [])
                + (['Atmosphere'] if self.atmos is not None else [])
                + ['SDO: ' + k for k in self.sdo])

    def coordinates(self, source):
        if source == 'Observation':
            return self.wave_obs
        if source == 'Fitted profiles':
            return self.wave_fit
        if source == 'Atmosphere':
            return self.tau
        if source.startswith('SDO:'):
            return self.wave_obs
        return np.array([0.])

    def image(self, source, time_index, index=0, stokes=0, parameter='temp',
              time_mode='utc', tolerance=60.):
        if source == 'Observation':
            return np.array(self.obs[time_index, stokes, :, :, index]), 'native intensity', ''
        if source == 'Fitted profiles':
            return np.array(self.fit['profiles'][time_index, :, :, index, stokes]), 'native intensity', ''
        if source == 'Atmosphere':
            scale, unit = {'temp': (1e-3, 'kK'), 'vlos': (1e-5, 'km/s'),
                           'vturb': (1e-5, 'km/s'), 'blong': (1., 'G')}.get(parameter, (1., 'native units'))
            return np.array(self.atmos[parameter][time_index, :, :, index]) * scale, unit, ''
        if time_mode != 'utc':
            raise ValueError('SDO matching now uses UTC and the saved manifest; nominal clocks are unsupported.')
        data, unit, detail, _ = self.sdo_view(source, time_index, tolerance)
        return data, unit, detail

    def _add_header(self, path, expected_index=None):
        header = fits.Header.fromtextfile(path)
        index = int(header['GRISIDX'])
        if expected_index is not None and expected_index != index:
            raise ValueError(f'GRIS WCS index disagrees with manifest: {path}')
        if index in self.gris_headers or not 0 <= index < self.nt:
            raise ValueError(f'Duplicate or out-of-range GRIS WCS index: {path}')
        if (header['NAXIS2'], header['NAXIS1']) != (self.ny, self.nx):
            raise ValueError(f'GRIS WCS shape differs from observation: {path}')
        if not WCS(header).has_celestial:
            raise ValueError(f'GRIS header has no solar WCS: {path}')
        self.gris_headers[index] = header

    def spatial_wcs(self, time_index):
        if self.gris_headers:
            return WCS(self.gris_headers[time_index])
        # An explicitly labelled relative coordinate system for standalone cubes.
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [(self.nx + 1) / 2, (self.ny + 1) / 2]
        wcs.wcs.crval = [0, 0]
        wcs.wcs.cdelt = [0.135, 0.135]
        wcs.wcs.cunit = ['arcsec', 'arcsec']
        wcs.wcs.ctype = ['LINEAR', 'LINEAR']
        return wcs

    @lru_cache(maxsize=3)
    def _sdo_map(self, path):
        return sunpy.map.Map(str(path))

    def matched_sdo(self, source, time_index, tolerance):
        channel = source.removeprefix('SDO: ')
        if self.sdo_matches is not None:
            frame = self.sdo_matches.get((channel, time_index), 'No manifest match for this frame/channel.')
            if isinstance(frame, str):
                raise ValueError(frame)
        else:
            frame = min(self.sdo[channel], key=lambda f: (abs((f.utc - self.times[time_index]).total_seconds()),
                                                         f.utc, f.path.name))
        delta = (frame.utc - self.times[time_index]).total_seconds()
        if abs(delta) > tolerance:
            raise ValueError(f'No SDO frame within {tolerance:g} s (selected: {delta:+.3f} s).')
        return frame, delta

    @staticmethod
    def sample_on_wcs(data, source_wcs, target_wcs, shape):
        y, x = np.indices(shape, dtype=float)
        sx, sy = pixel_to_pixel(target_wcs, source_wcs, x, y)
        return map_coordinates(np.asarray(data, dtype=float), [sy, sx], order=1,
                               mode='constant', cval=np.nan)

    @lru_cache(maxsize=12)
    def sdo_view(self, source, time_index, tolerance):
        if not self.gris_headers:
            raise ValueError('Load gris_wcs headers to locate GRIS on the full registered SDO image.')
        frame, delta = self.matched_sdo(source, time_index, tolerance)
        registered = self._sdo_map(frame.path)
        gris = sunpy.map.Map(np.zeros((self.ny, self.nx)), self.gris_headers[time_index])
        centre = gris.pixel_to_world((self.nx - 1) / 2 * u.pix, (self.ny - 1) / 2 * u.pix)
        # Display-only sampling: 200 × 0.25 arcsec = a 50 × 50 arcsec footprint.
        shape = (200, 200)
        header = sunpy.map.make_fitswcs_header(np.zeros(shape), centre,
                                               scale=[0.25, 0.25] * u.arcsec / u.pix)
        wcs = WCS(header)
        image = self.sample_on_wcs(registered.data, registered.wcs, wcs, shape)
        detail = (f'{frame.path.name}\nSource UTC: {frame.utc.isoformat()} | Δt {delta:+.3f} s'
                  '\n50″ × 50″ centred on GRIS; display resampling only.')
        return image, registered.meta.get('bunit', 'native units'), detail, wcs

    def display(self, source, time_index, index=0, stokes=0, parameter='temp', tolerance=60.):
        if not source.startswith('SDO:'):
            data, unit, detail = self.image(source, time_index, index, stokes, parameter)
            if not self.gris_headers:
                detail = 'Relative arcseconds from GRIS field centre (0.135 arcsec/sample).'
            return DisplayImage(data, unit, detail, self.spatial_wcs(time_index), bool(self.gris_headers))
        data, unit, detail, wcs = self.sdo_view(source, time_index, tolerance)
        gris_data = np.array(self.obs[time_index, 0, :, :, index])
        gris_wcs = self.spatial_wcs(time_index)
        overlay = self.sample_on_wcs(gris_data, gris_wcs, wcs, data.shape)
        return DisplayImage(data, unit, detail, wcs, True, gris_data, gris_wcs, overlay)

    def close(self):
        self._sdo_map.cache_clear()
        self.sdo_view.cache_clear()
        self.resources.close()
