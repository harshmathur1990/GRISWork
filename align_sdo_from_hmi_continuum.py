#!/usr/bin/env python3
"""Align other SDO channels to an already-aligned HMI continuum sequence.

``alignment_GUI_HMI.py`` writes each aligned HMI continuum image on the
ground-based image grid.  Consequently, the image shape and WCS in each of
those FITS files are a complete description of the required crop, rotation,
and pixel scale.  This script registers a raw SDO image and reprojects it onto
that grid.

By default, every source image sufficiently close in time to the aligned
continuum sequence is mapped to its nearest continuum frame for each of these
channels::

    HMI/Magnetogram
    AIA/171
    AIA/1600

The default paths reproduce the directory layout used in
``alignment_GUI_HMI.py``.  They can be changed from the command line.

Example
-------
python align_sdo_from_hmi_continuum.py \
    --raw-root /mnt/f/GRIS/SDO \
    --aligned-root /mnt/f/GRIS/aligned_SDO

The output is written below ``--aligned-root`` using the same relative channel
directories and the original source filenames.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


CHANNELS = ("HMI/Magnetogram", "AIA/171", "AIA/1600")

# This deliberately treats the clock in an HMI ``*_TAI`` filename as a
# nominal clock, rather than converting TAI to UTC.  alignment_GUI_HMI.py uses
# exactly this convention when it associates HMI images with ground frames.
HMI_TIME_RE = re.compile(r"\.(?P<date>\d{8})_(?P<time>\d{6})_TAI\.")
AIA_TIME_RE = re.compile(
    r"\.(?P<date>\d{4}-\d{2}-\d{2})T(?P<time>\d{6})Z\."
)


@dataclass(frozen=True)
class TimedFile:
    """A FITS path and its nominal sequence time."""

    time: datetime
    path: Path


@dataclass(frozen=True)
class Match:
    """A source image associated with an aligned continuum reference."""

    reference: TimedFile
    source: TimedFile
    delta_seconds: float


def nominal_time_from_name(path: Path) -> datetime:
    """Read an HMI or AIA nominal time from a standard SDO filename.

    Naive ``datetime`` objects are intentional: both clocks are compared as
    printed in the filenames, matching ``alignment_GUI_HMI.py``.
    """

    hmi_match = HMI_TIME_RE.search(path.name)
    if hmi_match:
        value = hmi_match.group("date") + hmi_match.group("time")
        return datetime.strptime(value, "%Y%m%d%H%M%S")

    aia_match = AIA_TIME_RE.search(path.name)
    if aia_match:
        value = aia_match.group("date") + aia_match.group("time")
        return datetime.strptime(value, "%Y-%m-%d%H%M%S")

    raise ValueError(f"Cannot extract an SDO time from filename: {path.name}")


def index_fits(directory: Path) -> list[TimedFile]:
    """Return all parseable FITS files in *directory*, sorted by time."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    indexed: list[TimedFile] = []
    rejected: list[str] = []
    for path in sorted(directory.glob("*.fits")):
        try:
            indexed.append(TimedFile(nominal_time_from_name(path), path))
        except ValueError:
            rejected.append(path.name)

    if rejected:
        print(
            f"Warning: ignored {len(rejected)} file(s) with unrecognized names "
            f"in {directory}",
            file=sys.stderr,
        )
    if not indexed:
        raise FileNotFoundError(f"No parseable FITS files found in: {directory}")

    return sorted(indexed, key=lambda item: (item.time, item.path.name))


def match_nearest(
    references: Sequence[TimedFile],
    sources: Sequence[TimedFile],
    max_delta_seconds: float,
) -> list[Match]:
    """Associate every in-range source with its nearest reference frame.

    The source and reference cadences need not be equal.  Iterating over the
    source sequence also ensures every output filename is unique.
    """

    matches: list[Match] = []
    for source in sources:
        reference = min(
            references,
            key=lambda item: (abs(item.time - source.time), item.time),
        )
        delta = abs((source.time - reference.time).total_seconds())
        if delta <= max_delta_seconds:
            matches.append(Match(reference, source, delta))

    if not matches:
        raise RuntimeError(
            f"No source frames are within {max_delta_seconds:g} s of the "
            "aligned continuum sequence"
        )

    return matches


def _load_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import the scientific packages only when processing is requested."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import numpy: {exc}. Install it into the Python "
            f"environment running this script ({sys.executable})."
        ) from exc

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import astropy.io.fits: {exc}. Install astropy into "
            f"the Python environment running this script ({sys.executable})."
        ) from exc

    try:
        import sunpy.map
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import sunpy.map: {exc}. Install SunPy (including its "
            f"map dependencies) into {sys.executable}."
        ) from exc

    try:
        from aiapy.calibrate import register
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import aiapy.calibrate.register: {exc}. Install aiapy "
            f"into {sys.executable}."
        ) from exc

    # Give a clear error up front instead of failing after the first large map
    # has been read.  SunPy imports without this optional reprojection package,
    # but Map.reproject_to needs it at runtime.
    try:
        import reproject  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import reproject: {exc}. Install reproject into the "
            f"Python environment running this script ({sys.executable})."
        ) from exc

    return np, sunpy.map, register, fits


def _copy_source_metadata(
    output_meta: Any,
    source_meta: Any,
    source_name: str,
    reference_name: str,
) -> Any:
    """Keep the target WCS while identifying the actual source observation."""

    # The reprojected map supplies the target spatial WCS.  Copy only
    # non-spatial observation metadata from the source, so no source CRPIX,
    # CDELT, PC, or CD keyword can corrupt the continuum alignment grid.
    source_keys = (
        "date-obs",
        "date_obs",
        "t_obs",
        "timesys",
        "telescop",
        "instrume",
        "detector",
        "wavelnth",
        "waveunit",
        "bunit",
        "content",
        "exptime",
        "quality",
        "lvl_num",
        "hglt_obs",
        "hgln_obs",
        "dsun_obs",
        "rsun_obs",
        "rsun_ref",
    )
    for key in source_keys:
        if key in source_meta:
            output_meta[key] = source_meta[key]

    output_meta["ALNMETH"] = "REGISTER+REPROJECT"
    output_meta["ALNREF"] = reference_name
    output_meta["SRCFILE"] = source_name
    return output_meta


def align_one(
    source_path: Path,
    reference_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
    do_register: bool,
) -> None:
    """Register and reproject one SDO image onto one continuum reference."""

    np, sunpy_map, register, fits = _load_dependencies()

    source_map = sunpy_map.Map(str(source_path))
    reference_map = sunpy_map.Map(str(reference_path))

    if source_map.data.ndim != 2 or reference_map.data.ndim != 2:
        raise ValueError(
            "Only two-dimensional image maps are supported: "
            f"source={source_map.data.shape}, reference={reference_map.data.shape}"
        )

    if do_register:
        try:
            source_map = register(source_map)
        except Exception as exc:
            raise RuntimeError(f"aiapy registration failed for {source_path}") from exc

    try:
        aligned_map = source_map.reproject_to(
            reference_map.wcs,
            shape_out=reference_map.data.shape,
            algorithm="interpolation",
            order="bilinear",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Reprojection failed for {source_path} using {reference_path}"
        ) from exc

    # Start from the metadata returned by reproject_to because it represents
    # the target grid.  Add source identity/time without replacing spatial WCS.
    output_meta = _copy_source_metadata(
        aligned_map.meta.copy(),
        source_map.meta,
        source_path.name,
        reference_path.name,
    )
    output_meta["naxis1"] = aligned_map.data.shape[1]
    output_meta["naxis2"] = aligned_map.data.shape[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(aligned_map.data, dtype=np.float32)
    fits.writeto(
        output_path,
        data,
        header=fits.Header(output_meta),
        overwrite=overwrite,
        output_verify="silentfix",
    )


def process_channel(
    channel: str,
    references: Sequence[TimedFile],
    raw_root: Path,
    aligned_root: Path,
    *,
    max_delta_seconds: float,
    overwrite: bool,
    do_register: bool,
    dry_run: bool,
) -> tuple[int, int]:
    """Align each temporally relevant source to its nearest reference."""

    sources = index_fits(raw_root / channel)
    matches = match_nearest(references, sources, max_delta_seconds)
    written = 0
    skipped = 0

    print(f"\n{channel}: {len(matches)} frame(s)")
    for number, match in enumerate(matches, start=1):
        output_path = aligned_root / channel / match.source.path.name
        message = (
            f"[{number:02d}/{len(matches):02d}] {match.source.path.name} -> "
            f"{match.reference.path.name} (dt={match.delta_seconds:.0f} s)"
        )

        if output_path.exists() and not overwrite:
            print(f"SKIP {message}")
            skipped += 1
            continue

        print(("PLAN " if dry_run else "DO   ") + message)
        if not dry_run:
            align_one(
                match.source.path,
                match.reference.path,
                output_path,
                overwrite=overwrite,
                do_register=do_register,
            )
            written += 1

    return written, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/f/GRIS/SDO"),
        help="Raw SDO root containing HMI/ and AIA/ (default: %(default)s)",
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=Path("/mnt/f/GRIS/aligned_SDO"),
        help=(
            "Aligned root containing HMI/Continuum; other channels are written "
            "below it (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=CHANNELS,
        default=list(CHANNELS),
        help="Channels to process (default: all three)",
    )
    parser.add_argument(
        "--max-time-delta",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Maximum allowed source/reference nominal time difference",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output FITS files that already exist",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip aiapy.calibrate.register (only for already-registered input)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print frame associations without reading or writing FITS data",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_time_delta < 0:
        raise ValueError("--max-time-delta must be non-negative")

    reference_dir = args.aligned_root / "HMI" / "Continuum"
    references = index_fits(reference_dir)
    print(f"Found {len(references)} aligned continuum reference frame(s)")

    total_written = 0
    total_skipped = 0
    for channel in args.channels:
        written, skipped = process_channel(
            channel,
            references,
            args.raw_root,
            args.aligned_root,
            max_delta_seconds=args.max_time_delta,
            overwrite=args.overwrite,
            do_register=not args.no_register,
            dry_run=args.dry_run,
        )
        total_written += written
        total_skipped += skipped

    if args.dry_run:
        print("\nDry run complete; no files were written.")
    else:
        print(
            f"\nComplete: wrote {total_written} file(s), "
            f"skipped {total_skipped} existing file(s)."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
