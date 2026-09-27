#!/usr/bin/env python3
"""Associate full registered SDO images with saved GRIS WCS headers.

Read HMI/Continuum/gris_wcs/*.hdr under --aligned-root (or --gris-wcs).
For each exact GRIS timestamp in serie_timestamps.csv, choose one nearest
observation per channel in UTC. Register only selected sources, once each, and
write <channel>/registered/<source-name>.fits. Write one
alignment.json at --aligned-root linking all channels to the GRIS headers.
No GRIS images, cropped images, reprojections, or differential rotations are
written. Source observation times and registered WCS remain intact.

Example:
    python align_sdo_from_hmi_continuum.py \\
        --raw-root /mn/stornext/d9/data/harshm/GRISData/SDO --aligned-root /mn/stornext/d9/data/harshm/GRISData/aligned_SDO
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from hmi_alignment import load_timestamps

CHANNELS = ("HMI/Continuum", "HMI/Magnetogram", "AIA/171", "AIA/1600", "AIA/304")
HMI_TIME_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})(?P<fraction>\.\d+)?_TAI", re.I)
HMI_FIDO_TIME_RE = re.compile(
    r"(?P<year>\d{4})[._-](?P<month>\d{2})[._-](?P<day>\d{2})[_T]"
    r"(?P<hour>\d{2})[:_]?(?P<minute>\d{2})[:_]?(?P<second>\d{2})"
    r"(?P<fraction>\.\d+)?_TAI", re.I)
AIA_TIME_RE = re.compile(
    r"(?P<year>\d{4})[-_](?P<month>\d{2})[-_](?P<day>\d{2})T"
    r"(?P<hour>\d{2})[_:]?(?P<minute>\d{2})[_:]?(?P<second>\d{2})"
    r"(?P<fraction>\.\d+)?Z", re.I)


@dataclass(frozen=True)
class TimedFile:
    time: datetime
    path: Path


@dataclass(frozen=True)
class GrisReference:
    frame_index: int
    time: datetime
    path: Path
    wcs_time: str
    shape: tuple[int, int]
    wavelength_index: int | None


def utc_time_from_name(path: Path) -> datetime:
    """Read JSOC/Fido names, converting explicitly labelled HMI TAI to UTC."""
    from astropy.time import Time

    match = HMI_TIME_RE.search(path.name)
    if match:
        value = datetime.strptime(match['date'] + match['time'], '%Y%m%d%H%M%S').isoformat()
        value += match['fraction'] or ''
        return Time(value, scale='tai').utc.to_datetime(timezone=timezone.utc)
    match = HMI_FIDO_TIME_RE.search(path.name)
    scale = 'tai'
    if match is None:
        match = AIA_TIME_RE.search(path.name)
        scale = 'utc'
    if match is None:
        raise ValueError(f'Cannot extract a UTC/TAI SDO time from filename: {path.name}')
    value = (f"{match['year']}-{match['month']}-{match['day']}T"
             f"{match['hour']}:{match['minute']}:{match['second']}{match['fraction'] or ''}")
    return Time(value, scale=scale).utc.to_datetime(timezone=timezone.utc)


def index_fits(directory: Path) -> list[TimedFile]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {directory}')
    indexed = []
    for path in sorted(directory.glob('*.fits')):
        try:
            indexed.append(TimedFile(utc_time_from_name(path), path))
        except ValueError as error:
            print(f'Warning: {error}', file=sys.stderr)
    if not indexed:
        raise FileNotFoundError(f'No parseable FITS files found in: {directory}')
    return sorted(indexed, key=lambda item: (item.time, item.path.name))


def load_gris_headers(directory: Path) -> list[GrisReference]:
    """Read header-only alignment outputs; no GRIS or continuum images needed."""
    from astropy.io import fits
    from astropy.time import Time
    from astropy.wcs import WCS

    references = []
    for path in sorted(directory.glob('*.hdr')):
        header = fits.Header.fromtextfile(path)
        try:
            index = int(header['GRISIDX'])
            stamp = datetime.fromisoformat(header['GRISDATE'].replace('Z', '+00:00'))
            shape = (int(header['NAXIS2']), int(header['NAXIS1']))
            wcs = WCS(header)
            if index < 0 or min(shape) <= 0 or not wcs.has_celestial or wcs.pixel_n_dim != 2:
                raise ValueError('Expected a non-negative GRISIDX and a two-dimensional solar WCS')
            if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
                raise ValueError('GRISDATE must explicitly use UTC')
            # Keep the coordinate reference time distinct from the GRIS exposure.
            wcs_time = Time(header['DATE-OBS'], scale=header.get('TIMESYS', 'UTC').lower()).utc.isot
            wave = int(header['GRISWAVE']) if 'GRISWAVE' in header else None
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f'Invalid GRIS header {path}: {error}') from error
        references.append(GrisReference(index, stamp.astimezone(timezone.utc), path, wcs_time, shape, wave))
    if not references:
        raise FileNotFoundError(f'No GRIS .hdr files found in: {directory}')
    references.sort(key=lambda item: item.frame_index)
    if len({ref.frame_index for ref in references}) != len(references):
        raise ValueError('Duplicate GRISIDX values in GRIS WCS headers')
    if any(b.time <= a.time for a, b in zip(references, references[1:])):
        raise ValueError('GRISDATE must increase with GRISIDX')
    return references


def match_nearest(reference: GrisReference, sources: Sequence[TimedFile],
                  max_delta_seconds: float | None) -> TimedFile | None:
    """Nearest source for this GRIS frame; shared sources are registered once."""
    if not sources:
        return None
    source = min(sources, key=lambda item: (abs(item.time - reference.time), item.time, item.path.name))
    if max_delta_seconds is not None and abs((source.time - reference.time).total_seconds()) > max_delta_seconds:
        return None
    return source


def register_one(source_path: Path, output_path: Path, *, overwrite: bool,
                 do_register: bool) -> None:
    """Save the full map, with its own observation WCS and time."""
    import sunpy.map
    source = sunpy.map.Map(str(source_path))
    if source.data.ndim != 2:
        raise ValueError(f'Only 2-D images are supported: {source_path}')
    if do_register:
        from aiapy.calibrate import register
        source = register(source)
    source.meta['SRCFILE'] = source_path.name
    source.meta['ALNMETH'] = 'REGISTER' if do_register else 'ALREADY_REGISTERED'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source.save(output_path, overwrite=overwrite)


def relative_path(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def process_channel(channel: str, references: Sequence[GrisReference], raw_root: Path,
                    aligned_root: Path, *, max_delta_seconds: float | None, overwrite: bool,
                    do_register: bool, dry_run: bool) -> tuple[dict[int, dict[str, Any]], int, int]:
    """Choose one source per GRIS time; register only the selected unique files."""
    records: dict[int, dict[str, Any]] = {}
    try:
        sources = index_fits(raw_root / channel)
    except FileNotFoundError as error:
        print(f'{channel}: {error}', file=sys.stderr)
        return {ref.frame_index: dict(status='unmatched', reason=str(error)) for ref in references}, 0, 0
    processed = {}
    written = skipped = 0
    for ref in references:
        source = match_nearest(ref, sources, max_delta_seconds)
        if source is None:
            records[ref.frame_index] = dict(status='unmatched', reason=f'No observation within {max_delta_seconds:g} s')
            print(f'{channel}: GRIS {ref.frame_index}: no match within {max_delta_seconds:g} s')
            continue
        if source.path not in processed:
            output = aligned_root / channel / 'registered' / source.path.name
            error = None
            if output.exists() and not overwrite:
                state = 'existing'
                skipped += 1
            elif dry_run:
                state = 'planned'
            else:
                try:
                    register_one(source.path, output, overwrite=overwrite, do_register=do_register)
                    state = 'written'
                    written += 1
                except Exception as exc:
                    state, error = 'failed', str(exc)
                    print(f'Failed to register {source.path}: {error}', file=sys.stderr)
            record = dict(status=state, source_sdo=str(source.path.resolve()),
                          timestamp_utc=source.time.isoformat(), timestamp_source='filename',
                          registered_sdo=relative_path(output, aligned_root) if state != 'failed' else None,
                          processing='existing_file' if state == 'existing' else
                                     ('register' if do_register else 'already_registered_input'))
            if error:
                record['reason'] = error
            processed[source.path] = record
        delta = (source.time - ref.time).total_seconds()
        records[ref.frame_index] = dict(processed[source.path], sdo_minus_gris_seconds=delta)
        print(f'{channel}: GRIS {ref.frame_index} at {ref.time.isoformat()} -> '
              f'{source.path.name} (SDO − GRIS={delta:+.3f} s)')
    return records, written, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=Path('/mn/stornext/d9/data/harshm/GRISData/SDO'),
                        help='Raw SDO root containing HMI/ and AIA/')
    parser.add_argument('--aligned-root', type=Path, default=Path('/mn/stornext/d9/data/harshm/GRISData/aligned_SDO'),
                        help='Output root for alignment.json and channel/registered/ directories')
    parser.add_argument('--gris-wcs', type=Path,
                        help='GRIS .hdr directory (default: <aligned-root>/HMI/Continuum/gris_wcs)')
    parser.add_argument('--timestamps', type=Path, default=Path(__file__).with_name('serie_timestamps.csv'),
                        help='CSV containing exact GRIS timestamps (default: serie_timestamps.csv beside script)')
    parser.add_argument('--channels', nargs='+', choices=CHANNELS, default=list(CHANNELS))
    parser.add_argument('--max-time-delta', type=float, default=None, metavar='SECONDS',
                        help='Optional maximum SDO/GRIS UTC offset; by default always select the nearest observation')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing registered FITS outputs')
    parser.add_argument('--no-register', action='store_true', help='Input is already registered; save full maps as supplied')
    parser.add_argument('--dry-run', action='store_true', help='Read the CSV, GRIS text headers, and SDO filenames; write nothing')
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_time_delta is not None and (not math.isfinite(args.max_time_delta) or args.max_time_delta < 0):
        raise ValueError('--max-time-delta must be finite and non-negative')
    headers_dir = args.gris_wcs or args.aligned_root / 'HMI' / 'Continuum' / 'gris_wcs'
    references = load_gris_headers(headers_dir)
    timestamps = load_timestamps(args.timestamps, len(references))
    if [ref.frame_index for ref in references] != list(range(len(timestamps))):
        raise ValueError('GRISIDX must cover each CSV series_index - 1 exactly once')
    # CSV exposure times determine temporal matches; the headers supply spatial WCS.
    references = [replace(ref, time=timestamps[ref.frame_index]) for ref in references]
    channels = list(dict.fromkeys(args.channels))
    frames = [dict(frame_index=ref.frame_index, series_index=ref.frame_index + 1,
                   timestamp_utc=ref.time.isoformat(), gris_header=relative_path(ref.path, args.aligned_root),
                   gris_shape=list(ref.shape), wavelength_index=ref.wavelength_index,
                   gris_wcs_reference_time_utc=ref.wcs_time, channels={}) for ref in references]
    written = skipped = problems = 0
    for channel in channels:
        records, n_written, n_skipped = process_channel(
            channel, references, args.raw_root, args.aligned_root,
            max_delta_seconds=args.max_time_delta, overwrite=args.overwrite,
            do_register=not args.no_register, dry_run=args.dry_run)
        written += n_written
        skipped += n_skipped
        for frame in frames:
            record = records[frame['frame_index']]
            frame['channels'][channel] = record
            problems += record['status'] in ('unmatched', 'failed')
    if args.dry_run:
        print(f'Dry run complete; {problems} unmatched/failed association(s); no files written.')
    else:
        args.aligned_root.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema_version=2, product='gris_sdo_registered_associations',
                        raw_root=str(args.raw_root.resolve()), gris_wcs=relative_path(headers_dir, args.aligned_root),
                        timestamps_csv=str(args.timestamps.resolve()),
                        max_time_delta_seconds=args.max_time_delta, channels=channels, frames=frames)
        manifest_path = args.aligned_root / 'alignment.json'
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(manifest_path)
        print(f'Wrote {written} full maps; reused {skipped}; {problems} unmatched/failed association(s).')
        print(f'Manifest: {manifest_path}')
    return 1 if problems else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(1)
