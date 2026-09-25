"""Lazy readers and time matching for the GRIS comparison viewer."""
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv

import h5py
import numpy as np
from astropy.io import fits
from astropy.time import Time
from align_sdo_from_hmi_continuum import nominal_time_from_name


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
    nominal: datetime
    utc: datetime
    reference: str
    unit: str


class DataStore:
    """Keep cubes on disk; only selected 2-D slices are materialized."""
    def __init__(self, observation, timestamps, atmosphere=None, profiles=None,
                 aligned_root=None, wave_start=8540.67304823, wave_step=0.0109907):
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
            if aligned_root:
                root = Path(aligned_root)
                if not root.is_dir():
                    raise ValueError(f'Aligned SDO directory does not exist: {root}')
                for path in sorted(root.rglob('*.fits')):
                    try:
                        with fits.open(path) as hdus:
                            hdu = next(h for h in hdus if h.header.get('NAXIS') == 2)
                            header = hdu.header
                            nominal = nominal_time_from_name(path)
                            # HMI filename clocks are TAI; AIA filename clocks are UTC.
                            scale = 'tai' if '_TAI' in path.name.upper() else 'utc'
                            utc = Time(nominal, scale=scale).utc.to_datetime()
                            frame = SDOFrame(path, nominal, utc, header.get('ALNREF', ''),
                                             header.get('BUNIT', 'native units'))
                        self.sdo.setdefault(path.parent.relative_to(root).as_posix(), []).append(frame)
                    except (ValueError, OSError, StopIteration) as exc:
                        self.warnings.append(f'Skipped {path.name}: {exc}')
                if not self.sdo:
                    self.warnings.append('No readable aligned SDO FITS images found.')
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
        return np.array([0.])

    def image(self, source, time_index, index=0, stokes=0, parameter='temp',
              time_mode='nominal', tolerance=60.):
        if source == 'Observation':
            return np.array(self.obs[time_index, stokes, :, :, index]), 'native intensity', ''
        if source == 'Fitted profiles':
            return np.array(self.fit['profiles'][time_index, :, :, index, stokes]), 'native intensity', ''
        if source == 'Atmosphere':
            scale, unit = {'temp': (1e-3, 'kK'), 'vlos': (1e-5, 'km/s'),
                           'vturb': (1e-5, 'km/s'), 'blong': (1., 'G')}.get(parameter, (1., 'native units'))
            return np.array(self.atmos[parameter][time_index, :, :, index]) * scale, unit, ''
        frames = self.sdo[source.removeprefix('SDO: ')]
        target = self.times[time_index]
        frame = min(frames, key=lambda f: (abs((getattr(f, time_mode) - target).total_seconds()),
                                          getattr(f, time_mode), f.path.name))
        delta = (getattr(frame, time_mode) - target).total_seconds()
        if abs(delta) > tolerance:
            raise ValueError(f'No SDO frame within {tolerance:g} s (nearest: {delta:+.1f} s).')
        with fits.open(frame.path, memmap=False) as hdus:
            data = np.array(next(h.data for h in hdus if h.header.get('NAXIS') == 2))
        if data.shape != (self.ny, self.nx):
            raise ValueError(f'SDO grid {data.shape} differs from observation {(self.ny, self.nx)}; no resampling applied.')
        detail = (f'{frame.path.name}\nSource {time_mode}: {getattr(frame, time_mode).isoformat()} '
                  f'| Δt {delta:+.1f} s\nAlignment reference: {frame.reference or "continuum itself"}')
        return data, frame.unit, detail

    def close(self):
        self.resources.close()
