#!/usr/bin/env python3
"""GUI for downloading selected SDO channels and aligning them to HMI.

The time range and target WCS grids are taken from the FITS files already in
``aligned_SDO/HMI/Continuum``.  Raw AIA data are requested with explicit DRMS
record-set queries, HMI data are requested through SunPy/Fido, and both are
then reprojected with the functions in
``align_sdo_from_hmi_continuum.py``.

Run with the same Python environment that contains SunPy and aiapy::

    python sdo_download_align_gui.py
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

from align_sdo_from_hmi_continuum import (
    TimedFile,
    align_one,
    index_fits,
    match_nearest,
    nominal_time_from_name,
)


Log = Callable[[str], None]

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class GuiTextStream:
    """Line-buffered stdout/stderr adapter for a GUI logging callback.

    TQDM/Parfive progress bars use carriage returns instead of newlines.  Those
    updates are throttled to keep the GUI event queue responsive during large
    downloads.
    """

    encoding = "utf-8"

    def __init__(self, emit: Log) -> None:
        self.emit = emit
        self.buffer = ""
        self.last_progress_emit = 0.0

    def write(self, text: str) -> int:
        if not text:
            return 0
        clean = ANSI_ESCAPE_RE.sub("", str(text))

        if "\r" in clean and "\n" not in clean:
            progress = clean.replace("\r", "").strip()
            now = time.monotonic()
            if progress and (now - self.last_progress_emit >= 1.0 or "100%" in progress):
                self.emit(progress)
                self.last_progress_emit = now
            return len(text)

        self.buffer += clean.replace("\r", "\n")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                self.emit(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self.buffer.strip():
            self.emit(self.buffer.rstrip())
        self.buffer = ""

    def isatty(self) -> bool:
        return False


class GuiLoggingHandler(logging.Handler):
    """Forward standard-library log records to the GUI callback."""

    def __init__(self, emit: Log) -> None:
        super().__init__()
        self.callback = emit
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.format(record))
        except Exception:
            self.handleError(record)


@contextlib.contextmanager
def route_console_output_to_gui(emit: Log) -> Any:
    """Temporarily route logs and console progress output into the GUI."""

    stream = GuiTextStream(emit)
    handler = GuiLoggingHandler(emit)
    loggers = [
        logging.getLogger(),
        logging.getLogger("drms"),
        logging.getLogger("sunpy"),
        logging.getLogger("parfive"),
    ]
    saved = [(logger, list(logger.handlers), logger.propagate) for logger in loggers]

    # Replace console handlers while the worker owns the pipeline.  Named
    # loggers do not propagate here, avoiding duplicate records at the root.
    for position, logger in enumerate(loggers):
        logger.handlers = [handler]
        if position:
            logger.propagate = False

    try:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            yield
    finally:
        stream.flush()
        for logger, handlers, propagate in saved:
            logger.handlers = handlers
            logger.propagate = propagate


class DiagnosticLog:
    """Tee pipeline messages to the GUI and a persistent UTF-8 text file."""

    def __init__(self, gui_log: Log, path: Path) -> None:
        self.gui_log = gui_log
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, message: str) -> None:
        text = str(message)
        self.gui_log(text)
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        for line in text.splitlines() or [""]:
            self.handle.write(f"{stamp} | {line}\n")

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True)
class Product:
    """One selectable, image-like JSOC product."""

    key: str
    label: str
    instrument: str
    raw_subdir: str
    aligned_subdir: str
    series: str
    wavelength: int | None = None
    native_cadence: int = 45
    segment: str | None = None
    note: str = ""


def _aia_product(wavelength: int, series: str, cadence: int, note: str = "") -> Product:
    return Product(
        key=f"aia_{wavelength}",
        label=f"AIA {wavelength} Å",
        instrument="AIA",
        raw_subdir=f"AIA/{wavelength}",
        aligned_subdir=f"AIA/{wavelength}",
        series=series,
        wavelength=wavelength,
        native_cadence=cadence,
        segment="image",
        note=note,
    )


AIA_PRODUCTS = (
    _aia_product(94, "aia.lev1_euv_12s", 12),
    _aia_product(131, "aia.lev1_euv_12s", 12),
    _aia_product(171, "aia.lev1_euv_12s", 12),
    _aia_product(193, "aia.lev1_euv_12s", 12),
    _aia_product(211, "aia.lev1_euv_12s", 12),
    _aia_product(304, "aia.lev1_euv_12s", 12),
    _aia_product(335, "aia.lev1_euv_12s", 12),
    _aia_product(1600, "aia.lev1_uv_24s", 24),
    _aia_product(1700, "aia.lev1_uv_24s", 24),
    _aia_product(
        4500,
        "aia.lev1_vis_1h",
        3600,
        "One-hour cadence; a short observing interval may contain no image.",
    ),
)

HMI_PRODUCTS = (
    Product(
        "hmi_continuum",
        "HMI continuum intensity",
        "HMI",
        "HMI/Continuum",
        "HMI/Continuum",
        "hmi.Ic_45s",
        note="Normally already present; also supplies the alignment WCS.",
    ),
    Product(
        "hmi_magnetogram",
        "HMI LOS magnetogram",
        "HMI",
        "HMI/Magnetogram",
        "HMI/Magnetogram",
        "hmi.M_45s",
    ),
    Product(
        "hmi_dopplergram",
        "HMI Dopplergram",
        "HMI",
        "HMI/Dopplergram",
        "HMI/Dopplergram",
        "hmi.V_45s",
    ),
    Product(
        "hmi_linewidth",
        "HMI line width",
        "HMI",
        "HMI/LineWidth",
        "HMI/LineWidth",
        "hmi.Lw_45s",
    ),
    Product(
        "hmi_linedepth",
        "HMI line depth",
        "HMI",
        "HMI/LineDepth",
        "HMI/LineDepth",
        "hmi.ld_45s",
    ),
)

ALL_PRODUCTS = AIA_PRODUCTS + HMI_PRODUCTS


def load_download_dependencies() -> tuple[Any, Any, Any]:
    """Load the network dependencies and name the exact missing component."""

    try:
        import astropy.units as u
    except ImportError as exc:
        raise RuntimeError(f"Could not import astropy.units: {exc}") from exc

    try:
        from sunpy.net import Fido, attrs as a
    except ImportError as exc:
        raise RuntimeError(
            "Could not import sunpy.net/Fido. Install SunPy's network extras "
            f"into {sys.executable}: {exc}"
        ) from exc

    try:
        import drms  # noqa: F401 - required by the JSOC client
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import drms (required for JSOC downloads): {exc}"
        ) from exc

    return u, Fido, a


def response_record_count(response: Any) -> int:
    """Return the number of records in a SunPy UnifiedResponse."""

    try:
        return sum(len(block) for block in response)
    except TypeError:
        return len(response)


def fits_validation_error(path: Path) -> str | None:
    """Return an explanation if *path* is not a complete readable FITS file.

    This checks every HDU boundary against the actual file size without loading
    the multi-megapixel image arrays into memory.
    """

    try:
        from astropy.io import fits

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with fits.open(
                path,
                mode="readonly",
                memmap=True,
                lazy_load_hdus=False,
            ) as hdus:
                if not hdus:
                    return "file contains no HDUs"
                hdus.verify("exception")
                file_size = os.path.getsize(path)
                has_image = False
                for index, hdu in enumerate(hdus):
                    if int(hdu.header.get("NAXIS", 0)) >= 2:
                        has_image = True
                    info = hdus.fileinfo(index)
                    if info is None:
                        continue
                    data_start = info.get("datLoc")
                    data_span = info.get("datSpan")
                    if data_start is not None and data_span is not None:
                        expected_end = int(data_start) + int(data_span)
                        if file_size < expected_end:
                            return (
                                f"truncated at {file_size} bytes; HDU {index} "
                                f"requires at least {expected_end} bytes"
                            )
                if not has_image:
                    return "file contains no two-dimensional image HDU"

            warning_text = " | ".join(str(item.message) for item in caught)
            if "truncated" in warning_text.lower():
                return warning_text
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"

    return None


def product_fits_files(product: Product, directory: Path) -> list[Path]:
    """Return local FITS candidates belonging to the requested image product."""

    paths = sorted(directory.glob("*.fits"))
    if product.instrument == "AIA":
        paths = [path for path in paths if ".spikes." not in path.name.lower()]
    return paths


def response_nominal_times(response: Any) -> set[Any]:
    """Extract nominal record clocks from a SunPy/JSOC response."""

    times: set[Any] = set()
    preferred_columns = (
        "T_REC",
        "T_OBS",
        "Start Time",
        "Start time",
        "start_time",
    )
    for block in response:
        column_names = list(getattr(block, "colnames", []))
        column = next((name for name in preferred_columns if name in column_names), None)
        if column is None:
            continue
        for value in block[column]:
            try:
                times.add(nominal_time_from_name(Path(str(value))))
            except ValueError:
                continue
    return times


def quarantine_file(path: Path, problem: str, log: Log) -> Path:
    """Move one invalid file aside without overwriting earlier quarantines."""

    candidate = path.with_name(path.name + ".invalid")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(path.name + f".invalid.{suffix}")
        suffix += 1
    path.replace(candidate)
    log(
        f"Moved invalid/incomplete download aside: {path.name} -> "
        f"{candidate.name} ({problem})"
    )
    return candidate


def assess_local_product(
    product: Product,
    destination: Path,
    window_start: Any,
    window_end: Any,
    log: Log,
    *,
    verbose: bool = False,
) -> tuple[list[Path], set[Any], list[Path]]:
    """Validate local product files within the requested nominal time window."""

    healthy_paths: list[Path] = []
    healthy_times: set[Any] = set()
    invalid_paths: list[Path] = []

    all_fits = sorted(destination.glob("*.fits"))
    candidates = product_fits_files(product, destination)
    if verbose:
        log(
            f"LOCAL INVENTORY: {len(candidates)} image FITS candidate(s) in "
            f"{destination}"
        )
        excluded = [path for path in all_fits if path not in candidates]
        for path in excluded:
            log(f"LOCAL EXCLUDED AUXILIARY FILE: {path}")

    for path in candidates:
        try:
            nominal_time = nominal_time_from_name(path)
        except ValueError:
            if verbose:
                log(f"LOCAL UNRECOGNIZED NAME: {path}")
            continue
        if nominal_time < window_start or nominal_time > window_end:
            if verbose:
                log(
                    f"LOCAL OUTSIDE INVENTORY DATE: time={nominal_time.isoformat()} "
                    f"size={path.stat().st_size} path={path}"
                )
            continue

        problem = fits_validation_error(path)
        if problem is not None:
            if verbose:
                log(
                    f"LOCAL INVALID: time={nominal_time.isoformat()} "
                    f"size={path.stat().st_size} path={path} reason={problem}"
                )
            invalid_paths.append(quarantine_file(path, problem, log))
            continue
        if verbose:
            log(
                f"LOCAL HEALTHY: time={nominal_time.isoformat()} "
                f"size={path.stat().st_size} path={path}"
            )
        healthy_paths.append(path)
        healthy_times.add(nominal_time)

    return healthy_paths, healthy_times, invalid_paths


def build_aia_drms_recordset(
    product: Product,
    references: Sequence[TimedFile],
    sample_seconds: int,
) -> str:
    """Build the explicit JSOC record-set query required for AIA exports.

    The observing sequence is rounded to its nominal duration and surrounded by
    a 5-minute leading and 10-minute trailing margin.  For the current 20-minute
    sequence this produces exactly ``09:03:00Z/35m@48s``.
    """

    if product.instrument != "AIA" or product.wavelength is None:
        raise ValueError("An AIA product with a wavelength is required")
    query_start = (references[0].time - timedelta(minutes=5)).replace(
        second=0, microsecond=0
    )
    observation_minutes = max(
        1,
        round((references[-1].time - references[0].time).total_seconds() / 60),
    )
    duration_minutes = observation_minutes + 15
    return (
        f"{product.series}[{query_start:%Y-%m-%dT%H:%M:%S}Z/"
        f"{duration_minutes}m@{sample_seconds}s]"
        f"[{product.wavelength}]{{image}}"
    )


def download_aia_with_drms(
    product: Product,
    references: Sequence[TimedFile],
    raw_root: Path,
    email: str,
    sample_seconds: int,
    log: Log,
) -> list[Path]:
    """Stage an explicit AIA DRMS export and download only missing files."""

    try:
        import drms
    except ImportError as exc:
        raise RuntimeError(f"Could not import drms: {exc}") from exc

    destination = raw_root / product.raw_subdir
    destination.mkdir(parents=True, exist_ok=True)
    recordset = build_aia_drms_recordset(product, references, sample_seconds)
    observation_start = references[0].time.replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    observation_end = references[-1].time.replace(
        hour=23, minute=59, second=59, microsecond=999999
    )

    log(f"DRMS RECORDSET QUERY: {recordset}")
    log(
        "DRMS EXPORT PARAMETERS: method='url', protocol='fits', "
        f"destination={str(destination)!r}"
    )
    client = drms.Client(email=email)
    request = client.export(
        recordset,
        method="url",
        protocol="fits",
        email=email,
    )
    log(
        f"DRMS EXPORT SUBMITTED: id={request.id!r}, initial_status={request.status}"
    )
    if not request.wait(sleep=10):
        raise RuntimeError(
            f"DRMS export did not complete successfully: id={request.id!r}, "
            f"status={request.status}"
        )
    log(
        f"DRMS EXPORT FINISHED: id={request.id!r}, status={request.status}, "
        f"request_url={request.request_url!r}"
    )

    urls = request.urls
    if urls is None or len(urls) == 0:
        healthy_paths, healthy_times, invalid = assess_local_product(
            product,
            destination,
            observation_start,
            observation_end,
            log,
            verbose=True,
        )
        estimated_count = max(
            1,
            round(
                (
                    (references[-1].time - references[0].time).total_seconds()
                    + 15 * 60
                )
                / sample_seconds
            ),
        )
        log(
            f"DRMS EMPTY EXPORT: local_healthy={len(healthy_times)}, "
            f"local_invalid={len(invalid)}, estimated_required={estimated_count}"
        )
        if len(healthy_times) >= estimated_count:
            log("Local AIA inventory is complete; download skipped.")
            return healthy_paths
        raise RuntimeError(
            f"DRMS returned no export URLs for {recordset}, and the local "
            "inventory is incomplete."
        )

    log(f"DRMS EXPORT FILE COUNT: {len(urls)}")
    healthy_paths, healthy_times, _ = assess_local_product(
        product,
        destination,
        observation_start,
        observation_end,
        log,
        verbose=True,
    )
    healthy_by_time = {
        nominal_time_from_name(path): path for path in healthy_paths
    }
    missing_indices: list[int] = []

    for ordinal, (index, row) in enumerate(urls.iterrows(), start=1):
        remote_filename = Path(str(row.get("filename", ""))).name
        remote_record = str(row.get("record", ""))
        remote_url = str(row.get("url", ""))
        local_path = destination / remote_filename if remote_filename else None
        local_ok = local_path is not None and local_path.exists()
        local_reason = "exact exported filename exists"

        if local_ok:
            problem = fits_validation_error(local_path)
            if problem is not None:
                quarantine_file(local_path, problem, log)
                local_ok = False
                local_reason = f"exact filename was invalid: {problem}"
        else:
            try:
                record_time = nominal_time_from_name(Path(remote_record))
            except ValueError:
                record_time = None
            if record_time is not None and record_time in healthy_by_time:
                local_ok = True
                local_path = healthy_by_time[record_time]
                local_reason = "healthy local timestamp match"
            else:
                local_reason = "no healthy exact-name or timestamp match"

        log(
            f"DRMS FILE [{ordinal:03d}/{len(urls):03d}]: index={index!r}, "
            f"record={remote_record!r}, filename={remote_filename!r}, "
            f"local_ok={local_ok}, local_path={str(local_path) if local_path else None!r}, "
            f"decision={local_reason!r}, url={remote_url!r}"
        )
        if not local_ok:
            missing_indices.append(index)

    if not missing_indices:
        log("All DRMS export files are already present and healthy; download skipped.")
        return healthy_paths

    log(
        f"DRMS DOWNLOAD PLAN: downloading {len(missing_indices)} of {len(urls)} "
        f"exported file(s), indices={missing_indices!r}"
    )
    result = request.download(
        str(destination),
        index=missing_indices,
        timeout=120,
    )
    log("DRMS DOWNLOAD RESULT:\n" + str(result))

    final_paths, _, invalid_after = assess_local_product(
        product,
        destination,
        observation_start,
        observation_end,
        log,
        verbose=True,
    )
    if invalid_after:
        raise RuntimeError(
            f"{len(invalid_after)} AIA file(s) were invalid after DRMS download."
        )
    return final_paths


def download_product(
    product: Product,
    references: Sequence[TimedFile],
    raw_root: Path,
    email: str,
    aia_sample_seconds: int,
    max_delta_seconds: float,
    overwrite: bool,
    log: Log,
) -> list[Path]:
    """Search JSOC and download one selected product."""

    if product.instrument == "AIA":
        return download_aia_with_drms(
            product,
            references,
            raw_root,
            email,
            aia_sample_seconds,
            log,
        )

    u, Fido, a = load_download_dependencies()
    destination = raw_root / product.raw_subdir
    destination.mkdir(parents=True, exist_ok=True)

    # Padding makes the nearest observation immediately outside either target
    # boundary available.  Times intentionally use the nominal filename clocks,
    # matching the association convention in alignment_GUI_HMI.py.
    pad = timedelta(seconds=max_delta_seconds)
    start = references[0].time - pad
    end = references[-1].time + pad
    # JSOC can expose a record timestamp in UTC while retaining a TAI clock in
    # the exported HMI filename.  Inventory the complete observation date so a
    # harmless time-scale representation difference cannot force Fido.fetch.
    inventory_start = start.replace(hour=0, minute=0, second=0, microsecond=0)
    inventory_end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    sample_seconds = (
        aia_sample_seconds if product.instrument == "AIA" else product.native_cadence
    )

    query_attrs: list[Any] = [
        a.Time(start, end),
        a.jsoc.Series(product.series),
        a.Sample(sample_seconds * u.s),
        a.jsoc.Notify(email),
    ]
    if product.wavelength is not None:
        query_attrs.append(a.Wavelength(product.wavelength * u.AA))
    if product.segment is not None:
        # AIA Level 1 also exposes a ``spikes`` segment containing cosmic-ray
        # detections.  Only the intensity image is suitable for registration.
        if product.instrument == "AIA" and product.segment == "image":
            # Use SunPy's registered attribute directly.  This is the form used
            # in the official JSOC AIA examples and is carried through to the
            # later export request even though segments are not displayed in
            # the search response table.
            query_attrs.append(a.jsoc.Segment.image)
        else:
            query_attrs.append(a.jsoc.Segment(product.segment))

    log(
        f"Searching {product.series} from {start.isoformat(sep=' ')} to "
        f"{end.isoformat(sep=' ')} (sample {sample_seconds} s)..."
    )
    log(
        "JSOC REQUEST: "
        f"series={product.series!r}, start={start.isoformat()!r}, "
        f"end={end.isoformat()!r}, sample_seconds={sample_seconds}, "
        f"wavelength_angstrom={product.wavelength!r}, "
        f"segment={product.segment!r}, destination={str(destination)!r}, "
        f"overwrite_raw=False"
    )
    log("JSOC QUERY ATTRIBUTES: " + " AND ".join(repr(item) for item in query_attrs))
    if product.segment is not None:
        log(f"JSOC export segment: {product.segment} only")
    response = Fido.search(*query_attrs)
    count = response_record_count(response)
    healthy_paths, healthy_times, invalid_before_fetch = assess_local_product(
        product,
        destination,
        inventory_start,
        inventory_end,
        log,
        verbose=True,
    )
    if count == 0:
        estimated_count = (
            len(references)
            if product.instrument == "HMI"
            else int((end - start).total_seconds() // sample_seconds) + 1
        )
        log(
            "JSOC EMPTY RESPONSE: the search returned zero records. "
            f"Cadence/time-span estimate={estimated_count}; local healthy "
            f"files on observation date={len(healthy_times)}; local invalid "
            f"files={len(invalid_before_fetch)}."
        )
        if len(healthy_times) >= estimated_count:
            log(
                "JSOC is unavailable or returned an empty result, but the local "
                "inventory covers the estimated request. Remote download skipped; "
                "continuing to alignment with validated local files."
            )
            return healthy_paths
        raise RuntimeError(
            f"JSOC returned no records for {product.label}, and only "
            f"{len(healthy_times)} healthy local file(s) were found; approximately "
            f"{estimated_count} are required. {product.note}".strip()
        )
    log("JSOC RESPONSE TABLE:\n" + str(response))
    for block_number, block in enumerate(response, start=1):
        log(
            f"JSOC RESPONSE BLOCK {block_number} COLUMNS: "
            f"{list(getattr(block, 'colnames', []))}"
        )
    expected_times = response_nominal_times(response)
    log(
        f"JSOC RESPONSE SUMMARY: providers={len(response)}, records={count}, "
        f"parsed_nominal_times={len(expected_times)}"
    )
    for position, record_time in enumerate(sorted(expected_times), start=1):
        log(
            f"JSOC EXPECTED [{position:03d}/{len(expected_times):03d}]: "
            f"{record_time.isoformat()}"
        )
    if expected_times:
        missing_times = expected_times - healthy_times
        exact_matches = expected_times & healthy_times
        log(
            f"JSOC returned {count} record(s). Local window: "
            f"{len(healthy_times)} healthy file(s), {len(exact_matches)} exact "
            f"timestamp match(es), {len(invalid_before_fetch)} invalid file(s), "
            f"{len(missing_times)} unmatched JSOC timestamp(s)."
        )
        if not missing_times:
            log("Every requested record is already available and healthy; download skipped.")
            return healthy_paths
        for position, record_time in enumerate(sorted(missing_times), start=1):
            log(
                f"MISSING RECORD [{position:03d}/{len(missing_times):03d}]: "
                f"{record_time.isoformat()}"
            )
    else:
        # This is a compatibility fallback for an unexpected response table
        # schema. Fido/Parfive will still skip existing destination files.
        log(
            f"JSOC returned {count} record(s), but their timestamps could not "
            "be read from the response; checking during fetch."
        )

    # Some JSOC table versions expose HMI TAI records as converted UTC values,
    # while exported filenames retain their nominal TAI clock.  In that case an
    # exact set comparison reports every local file as missing.  Record-count
    # parity on the same observation date is the safe fallback.
    if len(healthy_times) >= count:
        log(
            f"The observation date already contains {len(healthy_times)} healthy "
            f"local {product.label} file(s) for {count} JSOC record(s). Download "
            "skipped (JSOC/local timestamp representations differ)."
        )
        return healthy_paths

    log(f"Staging/downloading missing records to {destination}")
    log(
        "FETCH START: Fido.fetch will receive the complete JSOC response because "
        "JSOC does not support downloading a sliced response; overwrite=False "
        "should reuse paths whose exported filenames already exist."
    )

    downloaded = Fido.fetch(
        response,
        path=str(destination / "{file}"),
        # Raw files are immutable observations. Healthy local copies are never
        # replaced; invalid copies have already been quarantined above.
        overwrite=False,
    )
    errors = list(getattr(downloaded, "errors", []))
    if errors:
        log(f"Retrying {len(errors)} failed download(s)...")
        downloaded = Fido.fetch(downloaded)
        errors = list(getattr(downloaded, "errors", []))
    if errors:
        details = "; ".join(str(error) for error in errors[:3])
        raise RuntimeError(
            f"{len(errors)} download(s) failed for {product.label}: {details}"
        )

    # An interrupted network transfer can leave a partial file with its final
    # ``.fits`` name.  Quarantine it and run the same fetch once more; with
    # overwrite disabled, Parfive reuses all complete files and retrieves only
    # the now-missing records.
    _, downloaded_times, invalid = assess_local_product(
        product, destination, inventory_start, inventory_end, log
    )
    missing_after_fetch = expected_times - downloaded_times if expected_times else set()
    if len(downloaded_times) >= count:
        missing_after_fetch = set()
    if invalid or missing_after_fetch:
        log(
            f"Refetching {len(invalid)} invalid and "
            f"{len(missing_after_fetch)} missing FITS file(s)..."
        )
        repaired = Fido.fetch(
            response,
            path=str(destination / "{file}"),
            overwrite=False,
        )
        repair_errors = list(getattr(repaired, "errors", []))
        if repair_errors:
            details = "; ".join(str(error) for error in repair_errors[:3])
            raise RuntimeError(
                f"Repair download failed for {product.label}: {details}"
            )
        _, repaired_times, still_invalid = assess_local_product(
            product, destination, inventory_start, inventory_end, log
        )
        if still_invalid:
            raise RuntimeError(
                f"{len(still_invalid)} {product.label} FITS file(s) remained "
                "invalid after a second download attempt."
            )

        if expected_times:
            still_missing = expected_times - repaired_times
            if len(repaired_times) >= count:
                still_missing = set()
            if still_missing:
                raise RuntimeError(
                    f"{len(still_missing)} {product.label} record(s) are still "
                    "missing after the repair download."
                )

    paths = [Path(str(path)) for path in downloaded]
    for position, path in enumerate(paths, start=1):
        exists = path.exists()
        size = path.stat().st_size if exists else -1
        log(
            f"FETCH RESULT [{position:03d}/{len(paths):03d}]: "
            f"exists={exists} size={size} path={path}"
        )
    log(f"Download finished ({len(paths)} fetched or reused file(s)).")
    return paths


def align_product(
    product: Product,
    references: Sequence[TimedFile],
    raw_root: Path,
    aligned_root: Path,
    max_delta_seconds: float,
    overwrite: bool,
    log: Log,
) -> tuple[int, int]:
    """Align all relevant raw files for one product."""

    sources = index_fits(raw_root / product.raw_subdir)
    if product.instrument == "AIA":
        ignored = [source for source in sources if ".spikes." in source.path.name.lower()]
        sources = [source for source in sources if ".spikes." not in source.path.name.lower()]
        if ignored:
            log(
                f"Ignoring {len(ignored)} AIA spikes-segment file(s); only "
                "intensity images will be aligned."
            )
    matches = match_nearest(references, sources, max_delta_seconds)
    output_dir = aligned_root / product.aligned_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    log(f"Aligning {len(matches)} {product.label} frame(s)...")
    for number, match in enumerate(matches, start=1):
        output_path = output_dir / match.source.path.name
        if output_path.exists() and not overwrite:
            skipped += 1
            log(
                f"  [{number:02d}/{len(matches):02d}] exists, skipped: "
                f"{output_path.name}"
            )
            continue

        log(
            f"  [{number:02d}/{len(matches):02d}] {match.source.path.name} "
            f"(reference {match.reference.path.name}, dt={match.delta_seconds:.0f}s)"
        )
        align_one(
            match.source.path,
            match.reference.path,
            output_path,
            overwrite=overwrite,
            do_register=True,
        )
        written += 1

    return written, skipped


def _run_pipeline(
    products: Sequence[Product],
    raw_root: Path,
    aligned_root: Path,
    email: str,
    aia_sample_seconds: int,
    max_delta_seconds: float,
    overwrite: bool,
    log: Log,
) -> None:
    """Download and align all selected products serially."""

    references = index_fits(aligned_root / "HMI" / "Continuum")
    log(
        f"Using {len(references)} aligned HMI continuum references: "
        f"{references[0].time} to {references[-1].time}"
    )
    total_written = 0
    total_skipped = 0

    for position, product in enumerate(products, start=1):
        log("\n" + "=" * 72)
        log(f"Product {position}/{len(products)}: {product.label}")
        download_product(
            product,
            references,
            raw_root,
            email,
            aia_sample_seconds,
            max_delta_seconds,
            overwrite,
            log,
        )

        # The aligned continuum is the immutable reference itself.  Downloading
        # it can fill the raw directory, but it must not overwrite the reference
        # files by reprojecting them onto themselves.
        if product.key == "hmi_continuum":
            log("Continuum is the alignment reference; no alignment pass needed.")
            continue

        written, skipped = align_product(
            product,
            references,
            raw_root,
            aligned_root,
            max_delta_seconds,
            overwrite,
            log,
        )
        total_written += written
        total_skipped += skipped

    log("\n" + "=" * 72)
    log(
        f"Finished all products: wrote {total_written} aligned FITS file(s), "
        f"skipped {total_skipped} existing file(s)."
    )


def run_pipeline(
    products: Sequence[Product],
    raw_root: Path,
    aligned_root: Path,
    email: str,
    aia_sample_seconds: int,
    max_delta_seconds: float,
    overwrite: bool,
    log: Log,
) -> None:
    """Run the pipeline while saving a complete persistent diagnostic log."""

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%z")
    log_path = aligned_root / "logs" / f"sdo_download_align_{stamp}.log"
    diagnostic = DiagnosticLog(log, log_path)
    try:
        diagnostic("SDO DOWNLOAD/ALIGN DIAGNOSTIC LOG")
        diagnostic(f"Diagnostic log file: {log_path}")
        diagnostic(f"Python executable: {sys.executable}")
        diagnostic(f"Python version: {sys.version.replace(chr(10), ' ')}")
        diagnostic(f"Raw root: {raw_root}")
        diagnostic(f"Aligned root: {aligned_root}")
        diagnostic(
            "Selected products: " + ", ".join(product.label for product in products)
        )
        diagnostic(f"AIA sample seconds: {aia_sample_seconds}")
        diagnostic(f"Maximum alignment delta seconds: {max_delta_seconds}")
        diagnostic(f"Recreate aligned outputs: {overwrite}")
        _run_pipeline(
            products,
            raw_root,
            aligned_root,
            email,
            aia_sample_seconds,
            max_delta_seconds,
            overwrite,
            diagnostic,
        )
    except Exception:
        diagnostic("PIPELINE EXCEPTION:\n" + traceback.format_exc())
        raise
    finally:
        diagnostic.close()


def launch_gui(args: argparse.Namespace) -> int:
    """Create the Qt GUI lazily so ``--help`` works on headless systems."""

    try:
        from matplotlib.backends.qt_compat import QtCore, QtWidgets
    except ImportError as exc:
        raise SystemExit(
            "A Qt binding supported by Matplotlib is required (PyQt6, PySide6, "
            f"PyQt5, or PySide2). Original error: {exc}"
        ) from exc

    Signal = getattr(QtCore, "Signal", None) or QtCore.pyqtSignal
    Slot = getattr(QtCore, "Slot", None) or QtCore.pyqtSlot

    class PipelineWorker(QtCore.QObject):
        log_message = Signal(str)
        completed = Signal(bool, str)

        def __init__(self, pipeline_args: tuple[Any, ...]) -> None:
            super().__init__()
            self.pipeline_args = pipeline_args

        @Slot()
        def run(self) -> None:
            try:
                # Import SunPy/DRMS first because they configure their logging
                # handlers during import.  Replacing handlers after that makes
                # sure their staging messages go to the GUI rather than the
                # launching terminal.
                load_download_dependencies()
                with route_console_output_to_gui(self.log_message.emit):
                    run_pipeline(*self.pipeline_args, log=self.log_message.emit)
            except Exception as exc:
                self.log_message.emit("\nERROR\n" + traceback.format_exc())
                self.completed.emit(False, str(exc))
            else:
                self.completed.emit(True, "Download and alignment completed.")

    class DownloadAlignWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("SDO channel downloader and aligner")
            self.resize(980, 780)
            self.thread: Any = None
            self.worker: Any = None
            self.job_running = False
            self.product_boxes: dict[str, Any] = {}

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            outer = QtWidgets.QVBoxLayout(central)

            self.raw_edit = self._path_row(
                outer, "Raw SDO directory", str(args.raw_root)
            )
            self.aligned_edit = self._path_row(
                outer, "Aligned SDO directory", str(args.aligned_root)
            )

            email_row = QtWidgets.QHBoxLayout()
            email_row.addWidget(QtWidgets.QLabel("JSOC notification email"))
            self.email_edit = QtWidgets.QLineEdit(args.email)
            email_row.addWidget(self.email_edit, 1)
            outer.addLayout(email_row)

            settings = QtWidgets.QHBoxLayout()
            settings.addWidget(QtWidgets.QLabel("AIA sampling cadence (s)"))
            self.aia_sample = QtWidgets.QSpinBox()
            self.aia_sample.setRange(1, 86400)
            self.aia_sample.setValue(48)
            settings.addWidget(self.aia_sample)
            settings.addSpacing(18)
            settings.addWidget(QtWidgets.QLabel("Maximum reference offset (s)"))
            self.max_delta = QtWidgets.QDoubleSpinBox()
            self.max_delta.setRange(0, 86400)
            self.max_delta.setDecimals(1)
            self.max_delta.setValue(30)
            settings.addWidget(self.max_delta)
            settings.addSpacing(18)
            self.overwrite_box = QtWidgets.QCheckBox(
                "Recreate existing aligned files"
            )
            settings.addWidget(self.overwrite_box)
            settings.addStretch(1)
            outer.addLayout(settings)

            products_group = QtWidgets.QGroupBox("Available image channels")
            products_layout = QtWidgets.QHBoxLayout(products_group)
            products_layout.addWidget(self._product_group("AIA wavelengths", AIA_PRODUCTS))
            products_layout.addWidget(self._product_group("HMI observables", HMI_PRODUCTS))
            outer.addWidget(products_group)

            log_group = QtWidgets.QGroupBox("Progress")
            log_layout = QtWidgets.QVBoxLayout(log_group)
            self.log_text = QtWidgets.QPlainTextEdit()
            self.log_text.setReadOnly(True)
            log_layout.addWidget(self.log_text)
            outer.addWidget(log_group, 1)

            controls = QtWidgets.QHBoxLayout()
            self.status_label = QtWidgets.QLabel(
                "Choose channels, then click Download and align."
            )
            controls.addWidget(self.status_label, 1)
            self.start_button = QtWidgets.QPushButton("Download and align")
            self.start_button.clicked.connect(self.start)
            controls.addWidget(self.start_button)
            outer.addLayout(controls)

        def _path_row(self, outer: Any, label: str, initial: str) -> Any:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(label))
            edit = QtWidgets.QLineEdit(initial)
            row.addWidget(edit, 1)
            browse = QtWidgets.QPushButton("Browse…")
            browse.clicked.connect(lambda: self._browse(edit))
            row.addWidget(browse)
            outer.addLayout(row)
            return edit

        def _browse(self, edit: Any) -> None:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Choose directory", edit.text() or "/"
            )
            if chosen:
                edit.setText(chosen)

        def _product_group(self, title: str, products: Sequence[Product]) -> Any:
            group = QtWidgets.QGroupBox(title)
            layout = QtWidgets.QVBoxLayout(group)
            for product in products:
                text = product.label
                if product.note:
                    text += f" — {product.note}"
                box = QtWidgets.QCheckBox(text)
                self.product_boxes[product.key] = box
                layout.addWidget(box)
            layout.addStretch(1)
            return group

        def _arguments(self) -> tuple[Any, ...]:
            products = [
                product
                for product in ALL_PRODUCTS
                if self.product_boxes[product.key].isChecked()
            ]
            if not products:
                raise ValueError("Select at least one AIA or HMI channel.")

            email = self.email_edit.text().strip()
            if "@" not in email or email.startswith("@") or email.endswith("@"):
                raise ValueError("Enter the email address registered/used with JSOC.")

            raw_root = Path(self.raw_edit.text()).expanduser()
            aligned_root = Path(self.aligned_edit.text()).expanduser()
            index_fits(aligned_root / "HMI" / "Continuum")
            return (
                products,
                raw_root,
                aligned_root,
                email,
                self.aia_sample.value(),
                self.max_delta.value(),
                self.overwrite_box.isChecked(),
            )

        @Slot()
        def start(self) -> None:
            try:
                pipeline_args = self._arguments()
            except (ValueError, FileNotFoundError) as exc:
                QtWidgets.QMessageBox.critical(self, "Cannot start", str(exc))
                return

            self.log_text.clear()
            self.start_button.setEnabled(False)
            self.status_label.setText("Working… JSOC staging can take several minutes.")
            self.job_running = True
            self.thread = QtCore.QThread(self)
            self.worker = PipelineWorker(pipeline_args)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.log_message.connect(self._append_log)
            self.worker.completed.connect(self._finished)
            self.worker.completed.connect(self.thread.quit)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self._thread_stopped)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()

        @Slot(str)
        def _append_log(self, message: str) -> None:
            self.log_text.appendPlainText(message)

        @Slot(bool, str)
        def _finished(self, success: bool, message: str) -> None:
            if success:
                self.status_label.setText(message)
                QtWidgets.QMessageBox.information(self, "Complete", message)
            else:
                self.status_label.setText("Failed. See the progress log for details.")
                QtWidgets.QMessageBox.critical(
                    self, "Download/alignment failed", message
                )

        @Slot()
        def _thread_stopped(self) -> None:
            self.job_running = False
            self.start_button.setEnabled(True)
            self.worker = None
            self.thread = None

        def closeEvent(self, event: Any) -> None:
            if self.job_running:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Download still running",
                    "The JSOC download/alignment job is still running. Closing "
                    "the window now would terminate its worker thread and can "
                    "corrupt partial downloads. Please wait for completion; you "
                    "can minimize the window in the meantime.",
                )
                event.ignore()
                return
            event.accept()

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = DownloadAlignWindow()
    window.show()
    return int(application.exec())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/mnt/f/GRIS/SDO"),
        help="Initial raw SDO directory shown in the GUI",
    )
    parser.add_argument(
        "--aligned-root",
        type=Path,
        default=Path("/mnt/f/GRIS/aligned_SDO"),
        help="Initial aligned SDO directory shown in the GUI",
    )
    parser.add_argument(
        "--email",
        default="",
        help="Initial JSOC notification email shown in the GUI",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return launch_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
