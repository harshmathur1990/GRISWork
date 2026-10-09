#!/usr/bin/env python3
"""Make separate 4x and 2x GRIS-FOV movies at the original AIA cadence.

The original AIA FITS sequences under ``<raw-root>/AIA`` drive the movie. The
shortest-cadence AIA channel is the master clock; every other AIA channel and
HMI magnetogram is matched to its nearest exposure. One aligned GRIS WCS centre
(index 0 by default) anchors the entire movie and is differentially rotated to
each AIA/HMI exposure time. The anchor never switches between fitted headers,
so the tracked AIA scene does not jump because of frame-to-frame WCS-fit noise.
The native GRIS outline is fixed at 12x6 arcsec unless overridden. Plot axes
are relative to the selected field (e.g. 0, 5, 10, ... arcsec), never absolute
AIA coordinates. Every AIA and HMI source map is passed through
``aiapy.calibrate.register`` before spatial sampling. Unique
FITS files are registered once in a process pool using every available CPU by
default; both movie crops are cached from that single parallel pass.

Each output is a fixed 3x2 layout: up to five AIA channels followed by HMI
magnetogram as panel six. Every source is resampled onto the same per-frame
helioprojective grid, including HMI. The native GRIS FOV defaults to 12x6
arcsec: a 2x movie therefore samples centre X +/- 12 and centre Y +/- 6 arcsec,
while displaying axes 0..24 and 0..12 arcsec. All populated panels have the
same fixed integer tick values and angular
limits, so labels cannot make panels jump. Red and blue
contours show positive and negative HMI field on every panel. No colorbars are
drawn. AIA limits are fixed per channel over the full time series, using true
finite extrema so brightenings remain comparable and are not clipped.

Example:
    python make_aia_gris_fov_movie.py \
        --aligned-root /mn/stornext/d9/data/harshm/GRISData/aligned_SDO \
        --raw-root /mn/stornext/d9/data/harshm/GRISData/SDO \
        --output-prefix aia_hmi_gris

This writes ``aia_hmi_gris_4x.mp4`` and ``aia_hmi_gris_2x.mp4``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS

from align_sdo_from_hmi_continuum import TimedFile, index_fits


HMI_CHANNEL = "HMI/Magnetogram"
CADENCE_RE = re.compile(r"_(?P<seconds>\d+(?:\.\d+)?)s(?:\.|_)", re.I)


@dataclass(frozen=True)
class GrisFrame:
    index: int
    time: datetime
    header: Path


@dataclass(frozen=True)
class MovieFrame:
    index: int
    time: datetime
    aia: dict[str, TimedFile]
    magnetogram: TimedFile


@dataclass(frozen=True)
class SampledView:
    data: np.ndarray
    world_x_deg: np.ndarray
    world_y_deg: np.ndarray
    gris_x: np.ndarray
    gris_y: np.ndarray
    extent: tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceTask:
    path: Path
    centers: tuple[tuple[int, tuple[float, float]], ...]
    gris_fov: tuple[float, float]
    zooms: tuple[float, ...]
    pixel_scale: float


@dataclass(frozen=True)
class SourceProduct:
    path: Path
    samples: dict[tuple[int, float], np.ndarray]
    cmap_name: str


def display_channel(channel: str) -> str:
    return channel.split("/", 1)[1] if channel.startswith("AIA/") else channel


def canonical_channel(channel: str) -> str:
    return channel if channel.startswith("AIA/") else f"AIA/{channel}"


def nearest(time: datetime, files: Sequence[TimedFile]) -> TimedFile:
    if not files:
        raise ValueError("Cannot match an empty observation sequence")
    return min(files, key=lambda item: (abs(item.time - time), item.time, item.path.name))


def load_manifest(aligned_root: Path) -> tuple[dict, list[GrisFrame]]:
    manifest_path = aligned_root / "alignment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing alignment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest.get("frames", []), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError("alignment.json contains no GRIS frames")
    indices = [int(row["frame_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Manifest frame_index values must be consecutive from zero")
    frames = []
    for row in rows:
        index = int(row["frame_index"])
        stamp = datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError(f"GRIS frame {index} timestamp does not specify UTC")
        relative = row.get("gris_header", f"HMI/Continuum/gris_wcs/gris_{index:04d}.hdr")
        header = aligned_root / relative
        if not header.is_file():
            raise FileNotFoundError(f"Missing GRIS WCS header: {header}")
        frames.append(GrisFrame(index, stamp, header))
    return manifest, frames


def discover_aia(raw_root: Path, requested: Sequence[str] | None,
                 start: datetime, end: datetime) -> tuple[list[str], dict[str, list[TimedFile]]]:
    aia_root = raw_root / "AIA"
    if not aia_root.is_dir():
        raise FileNotFoundError(f"Missing original AIA root: {aia_root}")
    if requested:
        channels = list(dict.fromkeys(canonical_channel(value) for value in requested))
    else:
        channels = [f"AIA/{path.name}" for path in aia_root.iterdir()
                    if path.is_dir() and any(path.glob("*.fits"))]
        channels.sort(key=lambda value: (int(display_channel(value))
                                         if display_channel(value).isdigit() else math.inf,
                                         display_channel(value)))
    if not channels:
        raise ValueError(f"No original AIA channel directories found below {aia_root}")
    if len(channels) > 5:
        raise ValueError("A 3x2 layout allows five AIA channels; select five with --channels")

    indexed: dict[str, list[TimedFile]] = {}
    for channel in channels:
        sequence = [item for item in index_fits(raw_root / channel) if start <= item.time <= end]
        if not sequence:
            raise ValueError(f"No {channel} images fall inside the GRIS time interval")
        indexed[channel] = sequence
    return channels, indexed


def nominal_cadence(files: Sequence[TimedFile]) -> float:
    matches = [CADENCE_RE.search(item.path.name) for item in files]
    values = [float(match.group("seconds")) for match in matches if match]
    if values:
        return min(values)
    if len(files) > 1:
        return float(np.median([(b.time - a.time).total_seconds()
                                for a, b in zip(files, files[1:])]))
    return math.inf


def choose_cadence_channel(channels: Sequence[str], indexed: dict[str, list[TimedFile]],
                           requested: str | None) -> str:
    if requested:
        selected = canonical_channel(requested)
        if selected not in indexed:
            raise ValueError(f"Cadence channel is not selected/available: {selected}")
        return selected
    return min(channels, key=lambda channel: (nominal_cadence(indexed[channel]),
                                               -len(indexed[channel]), channel))


def build_movie_frames(master: str, channels: Sequence[str],
                       indexed: dict[str, list[TimedFile]],
                       magnetograms: Sequence[TimedFile]) -> list[MovieFrame]:
    frames = []
    for index, clock in enumerate(indexed[master]):
        frames.append(MovieFrame(index, clock.time,
                                 {channel: nearest(clock.time, indexed[channel]) for channel in channels},
                                 nearest(clock.time, magnetograms)))
    return frames


def read_gris_wcs(path: Path) -> tuple[WCS, tuple[int, int]]:
    header = fits.Header.fromtextfile(path)
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    wcs = WCS(header)
    if min(shape) <= 0 or not wcs.has_celestial:
        raise ValueError(f"Invalid GRIS WCS header: {path}")
    return wcs, shape


def rotated_gris_center(reference: GrisFrame, target_time: datetime) -> tuple[float, float]:
    """Rotate one fixed aligned GRIS centre to an SDO exposure time."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from sunpy.coordinates import Helioprojective, get_body_heliographic_stonyhurst
    from sunpy.physics.differential_rotation import solar_rotate_coordinate

    header = fits.Header.fromtextfile(reference.header)
    wcs = WCS(header)
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    world_x, world_y = wcs.celestial.pixel_to_world_values(
        (nx - 1) / 2, (ny - 1) / 2)
    world_x = ((float(world_x) + 180.0) % 360.0) - 180.0
    reference_time = header.get("DATE-OBS", reference.time.isoformat())
    coordinate = SkyCoord(world_x * u.deg, float(world_y) * u.deg,
                          frame=Helioprojective(observer="earth", obstime=reference_time))
    new_observer = get_body_heliographic_stonyhurst("earth", Time(target_time))
    rotated = solar_rotate_coordinate(coordinate, observer=new_observer)
    center = float(rotated.Tx.to_value(u.arcsec)), float(rotated.Ty.to_value(u.arcsec))
    if not np.all(np.isfinite(center)):
        raise ValueError(f"Differential rotation produced a non-finite centre at {target_time}")
    return center


def load_registered_map(path: Path):
    """Load and register one full-disk AIA or HMI level-1 map."""
    import sunpy.map
    from aiapy.calibrate import register

    result = sunpy.map.Map(str(path))
    result = register(result)
    if result.data.ndim != 2:
        raise ValueError(f"Expected a 2-D image: {path}")
    return result


def fixed_grid(center: tuple[float, float], gris_fov: tuple[float, float],
               zoom: float, pixel_scale: float) -> SampledView:
    """Create a world-coordinate sampling grid with fixed relative plot axes."""
    center_x, center_y = center
    gris_width, gris_height = gris_fov
    half_width = gris_width * zoom / 2
    half_height = gris_height * zoom / 2
    left, right = center_x - half_width, center_x + half_width
    bottom, top = center_y - half_height, center_y + half_height
    nx = max(2, int(math.ceil((right - left) / pixel_scale)))
    ny = max(2, int(math.ceil((top - bottom) / pixel_scale)))
    dx, dy = (right - left) / nx, (top - bottom) / ny
    x_arcsec = left + (np.arange(nx) + .5) * dx
    y_arcsec = bottom + (np.arange(ny) + .5) * dy
    world_x, world_y = np.meshgrid(x_arcsec / 3600.0, y_arcsec / 3600.0)
    gris_left, gris_right = half_width - gris_width / 2, half_width + gris_width / 2
    gris_bottom, gris_top = half_height - gris_height / 2, half_height + gris_height / 2
    gris_x = np.array([gris_left, gris_right, gris_right, gris_left, gris_left])
    gris_y = np.array([gris_bottom, gris_bottom, gris_top, gris_top, gris_bottom])
    return SampledView(np.empty((ny, nx)), world_x, world_y, gris_x, gris_y,
                       (0.0, right - left, 0.0, top - bottom))


def sample_to_view(source, view: SampledView) -> np.ndarray:
    """Sample a map onto a frame's differentially rotated world grid."""
    from scipy.ndimage import map_coordinates

    source_x, source_y = source.wcs.celestial.world_to_pixel_values(
        view.world_x_deg, view.world_y_deg)
    return map_coordinates(np.asarray(source.data, dtype=float), [source_y, source_x],
                           order=1, mode="constant", cval=np.nan)


def register_and_sample(task: SourceTask) -> SourceProduct:
    """Worker process: register one unique full-disk file and sample all FOVs."""
    source = load_registered_map(task.path)
    samples = {
        (frame_index, zoom): np.asarray(sample_to_view(
            source, fixed_grid(center, task.gris_fov, zoom, task.pixel_scale)),
            dtype=np.float32)
        for frame_index, center in task.centers
        for zoom in task.zooms
    }
    cmap = source.plot_settings.get("cmap", "gray")
    return SourceProduct(task.path, samples, getattr(cmap, "name", str(cmap)))


def available_cpus() -> int:
    """Respect scheduler/OS affinity where available, otherwise use all CPUs."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def preprocess_sources(frames: Sequence[MovieFrame],
                       centers: dict[tuple[int, Path], tuple[float, float]],
                       gris_fov: tuple[float, float], zooms: Sequence[float],
                       pixel_scale: float, workers: int) -> dict[Path, SourceProduct]:
    """Register every unique source once in parallel and retain only small crops."""
    requests: dict[Path, set[int]] = {}
    for frame in frames:
        requests.setdefault(frame.magnetogram.path, set()).add(frame.index)
        for source in frame.aia.values():
            requests.setdefault(source.path, set()).add(frame.index)
    tasks = [SourceTask(path, tuple((index, centers[index, path]) for index in sorted(indices)),
                        gris_fov, tuple(zooms), pixel_scale)
             for path, indices in sorted(requests.items())]
    worker_count = min(workers, len(tasks))
    print(f"Parallel preprocessing: {len(tasks)} unique FITS files on {worker_count} workers")
    products: dict[Path, SourceProduct] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(register_and_sample, task): task.path for task in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                product = future.result()
            except Exception as exc:
                raise RuntimeError(f"Parallel registration failed for {path}: {exc}") from exc
            products[product.path] = product
            print(f"Registering and sampling: {number}/{len(tasks)}", end="\r", flush=True)
    print()
    return products


def limits_from_products(channels: Sequence[str], frames: Sequence[MovieFrame],
                         products: dict[Path, SourceProduct], largest_zoom: float
                         ) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    """Compute fixed limits from already registered and sampled arrays."""
    limits: dict[str, tuple[float, float]] = {}
    cmaps: dict[str, str] = {}
    for channel in channels:
        paths = {frame.aia[channel].path for frame in frames}
        arrays = [products[frame.aia[channel].path].samples[frame.index, largest_zoom]
                  for frame in frames]
        finite_arrays = [values[np.isfinite(values)] for values in arrays]
        finite_arrays = [values for values in finite_arrays if values.size]
        if not finite_arrays:
            raise ValueError(f"No finite pixels found for {channel}")
        low = min(float(np.min(values)) for values in finite_arrays)
        high = max(float(np.max(values)) for values in finite_arrays)
        if low == high:
            pad = max(abs(low) * 1e-6, 1e-12)
            low, high = low - pad, high + pad
        limits[channel] = (low, high)
        cmaps[channel] = products[next(iter(paths))].cmap_name
        print(f"{channel}: fixed full-series limits [{low:.8g}, {high:.8g}]")

    hmi_paths = {frame.magnetogram.path for frame in frames}
    hmi_arrays = [products[frame.magnetogram.path].samples[frame.index, largest_zoom]
                  for frame in frames]
    finite_hmi = [values[np.isfinite(values)] for values in hmi_arrays]
    finite_hmi = [values for values in finite_hmi if values.size]
    if not finite_hmi:
        raise ValueError("No finite HMI magnetogram pixels found")
    hmi_absmax = max(float(np.max(np.abs(values))) for values in finite_hmi)
    if not math.isfinite(hmi_absmax) or hmi_absmax <= 0:
        raise ValueError("No finite non-zero HMI magnetogram pixels found")
    limits[HMI_CHANNEL] = (-hmi_absmax, hmi_absmax)
    cmaps[HMI_CHANNEL] = "gray"
    print(f"{HMI_CHANNEL}: fixed symmetric limits [{-hmi_absmax:.8g}, {hmi_absmax:.8g}]")
    return limits, cmaps


def remove_contour(contour) -> None:
    if contour is None:
        return
    try:
        contour.remove()
    except AttributeError:
        for collection in contour.collections:
            collection.remove()


def output_path(prefix: Path, zoom: float, movie_format: str) -> Path:
    base = prefix.with_suffix("") if prefix.suffix.lower() in (".mp4", ".gif") else prefix
    return base.with_name(f"{base.name}_{zoom:g}x.{movie_format}")


def render_movie(channels: Sequence[str], frames: Sequence[MovieFrame], output: Path,
                 zoom: float, gris_fov: tuple[float, float],
                 pixel_scale: float, limits: dict[str, tuple[float, float]],
                 cmaps: dict[str, str], products: dict[Path, SourceProduct],
                 fps: float, dpi: int, asinh_a: float, pore_field: float,
                 tick_step: int) -> None:
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import sunpy.visualization.colormaps  # Register AIA colormap names with Matplotlib.
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    # Rendering uses only relative coordinates; source-specific absolute WCS
    # centers were already applied during parallel preprocessing.
    common_view = fixed_grid((0.0, 0.0), gris_fov, zoom, pixel_scale)
    left, right, bottom, top = common_view.extent
    width, height = right - left, top - bottom
    panels = list(channels) + [None] * (5 - len(channels)) + [HMI_CHANNEL]
    x_ticks = np.arange(0, math.floor(width) + 1, tick_step, dtype=int)
    y_ticks = np.arange(0, math.floor(height) + 1, tick_step, dtype=int)
    norms = {channel: ImageNormalize(vmin=limits[channel][0], vmax=limits[channel][1],
                                     stretch=AsinhStretch(asinh_a), clip=True)
             for channel in channels}
    norms[HMI_CHANNEL] = Normalize(*limits[HMI_CHANNEL], clip=True)

    fig, axes_array = plt.subplots(2, 3, figsize=(12.0, 7.8), constrained_layout=True)
    axes = list(axes_array.flat)
    images: dict[int, object] = {}
    boundaries: dict[int, object] = {}
    magnetic_contours: dict[int, object] = {}
    cached_number: int | None = None
    cached: dict[int, tuple[SampledView, np.ndarray]] = {}

    def views_for(frame_number: int) -> dict[int, tuple[SampledView, np.ndarray]]:
        nonlocal cached_number, cached
        frame = frames[frame_number]
        magnetic_field = products[frame.magnetogram.path].samples[frame.index, zoom]
        result: dict[int, tuple[SampledView, np.ndarray]] = {}
        for panel_index, channel in enumerate(panels):
            if channel is None:
                continue
            if channel == HMI_CHANNEL:
                data = magnetic_field
            else:
                data = products[frame.aia[channel].path].samples[frame.index, zoom]
            # All six panels have identical per-frame sampling, shape, relative extent,
            # GRIS outline coordinates, and HMI contour pixels.
            result[panel_index] = replace(common_view, data=data), magnetic_field
        cached_number, cached = frame_number, result
        return result

    initial = views_for(0)
    for panel_index, (ax, channel) in enumerate(zip(axes, panels)):
        if channel is None:
            ax.set_axis_off()
            continue
        view, field = initial[panel_index]
        image = ax.imshow(view.data, origin="lower", extent=view.extent,
                          cmap=cmaps[channel], norm=norms[channel], interpolation="nearest")
        boundary, = ax.plot(view.gris_x, view.gris_y, color="cyan", lw=1.8, zorder=6)
        boundary.set_path_effects([path_effects.Stroke(linewidth=3.2, foreground="black"),
                                   path_effects.Normal()])
        contour = ax.contour(field, levels=[-pore_field, pore_field],
                             colors=["dodgerblue", "red"], linewidths=1.25,
                             origin="lower", extent=view.extent, zorder=5)
        ax.set_xlim(view.extent[0], view.extent[1])
        ax.set_ylim(view.extent[2], view.extent[3])
        ax.set_title("HMI magnetogram" if channel == HMI_CHANNEL
                     else f"AIA {display_channel(channel)} Å")
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        if panel_index >= 3:
            ax.set_xlabel("X [arcsec]")
        if panel_index % 3 == 0:
            ax.set_ylabel("Y [arcsec]")
        images[panel_index] = image
        boundaries[panel_index] = boundary
        magnetic_contours[panel_index] = contour

    legend_handles = [
        Line2D([0], [0], color="cyan", lw=2, label="GRIS FOV"),
        Line2D([0], [0], color="red", lw=1.5, label=f"+{pore_field:g} G"),
        Line2D([0], [0], color="dodgerblue", lw=1.5, label=f"−{pore_field:g} G"),
    ]
    axes[5].legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=.75)
    title = fig.suptitle("")

    def update(frame_number: int):
        frame = frames[frame_number]
        current = cached if cached_number == frame_number else views_for(frame_number)
        returned = [title]
        for panel_index, channel in enumerate(panels):
            if channel is None:
                continue
            view, field = current[panel_index]
            images[panel_index].set_data(view.data)
            boundaries[panel_index].set_data(view.gris_x, view.gris_y)
            remove_contour(magnetic_contours[panel_index])
            magnetic_contours[panel_index] = axes[panel_index].contour(
                field, levels=[-pore_field, pore_field], colors=["dodgerblue", "red"],
                linewidths=1.25, origin="lower", extent=view.extent, zorder=5)
            returned.extend([images[panel_index], boundaries[panel_index]])
        title.set_text(f"AIA cadence  {frame.index + 1:03d}/{len(frames):03d}  "
                       f"{frame.time.isoformat()}  |  {zoom:g}× GRIS FOV")
        print(f"Rendering {output.name}: {frame_number + 1}/{len(frames)}",
              end="\r", flush=True)
        return returned

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False, repeat=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for MP4 output; use --format gif otherwise")
        writer = FFMpegWriter(fps=fps, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-threads", "0"])
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"\nWrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path,
                        default=Path("/mn/stornext/d9/data/harshm/GRISData/aligned_SDO"),
                        help="Root containing alignment.json and GRIS WCS headers")
    parser.add_argument("--raw-root", type=Path,
                        help="Original SDO root containing AIA/ and HMI/ (default: raw_root in manifest)")
    parser.add_argument("--output-prefix", type=Path, default=Path("aia_hmi_gris"),
                        help="Prefix for separate _4x and _2x movies")
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--channels", nargs="+", metavar="WAVELENGTH",
                        help="Up to five AIA channels; default: all original AIA channel folders")
    parser.add_argument("--cadence-channel", metavar="WAVELENGTH",
                        help="AIA channel whose original timestamps drive the movie; default: shortest cadence")
    parser.add_argument("--fov", type=float, nargs="+", default=[4.0, 2.0],
                        help="Separate GRIS FOV multipliers (default: 4 2)")
    parser.add_argument("--gris-fov", type=float, nargs=2, default=[12.0, 6.0],
                        metavar=("WIDTH", "HEIGHT"),
                        help="Native GRIS FOV in arcsec (default: 12 6)")
    parser.add_argument("--reference-gris-index", type=int, default=0, metavar="INDEX",
                        help="Single GRIS WCS anchor used for tracking (default: 0)")
    parser.add_argument("--pixel-scale", type=float, default=0.6, metavar="ARCSEC",
                        help="Fixed movie-grid sampling (default: 0.6 arcsec/pixel)")
    parser.add_argument("--tick-step", type=int, default=5, metavar="ARCSEC",
                        help="Integer spacing of relative X/Y ticks (default: 5 arcsec)")
    parser.add_argument("--pore-field", type=float, default=500.0, metavar="GAUSS",
                        help="Absolute HMI levels for negative/positive pore contours (default: 500 G)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel registration workers (default: 0 = all available CPUs)")
    parser.add_argument("--asinh-a", type=float, default=0.01,
                        help="AIA asinh stretch transition parameter (default: 0.01)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for option, value in (("--fps", args.fps), ("--pore-field", args.pore_field),
                          ("--asinh-a", args.asinh_a), ("--pixel-scale", args.pixel_scale)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{option} must be finite and positive")
    if args.dpi <= 0 or not args.fov or any(not math.isfinite(v) or v <= 0 for v in args.fov):
        raise ValueError("--dpi and every --fov value must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative (0 means all available CPUs)")
    if args.tick_step <= 0:
        raise ValueError("--tick-step must be a positive integer")
    if any(not math.isfinite(v) or v <= 0 for v in args.gris_fov):
        raise ValueError("Both --gris-fov dimensions must be finite and positive")
    manifest, gris_frames = load_manifest(args.aligned_root)
    if not 0 <= args.reference_gris_index < len(gris_frames):
        raise ValueError(f"--reference-gris-index must be between 0 and {len(gris_frames) - 1}")
    reference_gris = gris_frames[args.reference_gris_index]
    saved_raw_root = manifest.get("raw_root")
    if args.raw_root is None and not saved_raw_root:
        raise ValueError("alignment.json has no raw_root; pass --raw-root explicitly")
    raw_root = args.raw_root or Path(saved_raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError("Original SDO root is unavailable; pass --raw-root explicitly")
    channels, indexed = discover_aia(raw_root, args.channels,
                                     gris_frames[0].time, gris_frames[-1].time)
    master = choose_cadence_channel(channels, indexed, args.cadence_channel)
    magnetograms = index_fits(raw_root / HMI_CHANNEL)
    frames = build_movie_frames(master, channels, indexed, magnetograms)
    if len(channels) < 5:
        print(f"Warning: found {len(channels)} AIA channels; unused AIA slots will be blank.",
              file=sys.stderr)
    print(f"Master AIA cadence: {master} ({len(frames)} original exposures); "
          f"panels: {', '.join(channels)}, {HMI_CHANNEL}")

    native_fov = tuple(args.gris_fov)
    centers: dict[tuple[int, Path], tuple[float, float]] = {}
    for frame in frames:
        sources = [*frame.aia.values(), frame.magnetogram]
        for source in sources:
            centers[frame.index, source.path] = rotated_gris_center(reference_gris, source.time)
    print(f"Native GRIS FOV={native_fov[0]:g}×{native_fov[1]:g} arcsec; "
          f"tracking from fixed GRIS WCS index {reference_gris.index} with differential rotation.")
    largest_zoom = max(args.fov)
    workers = args.workers or available_cpus()
    print("Registering every unique AIA and HMI source frame once with "
          "aiapy.calibrate.register().")
    products = preprocess_sources(frames, centers, native_fov, args.fov,
                                  args.pixel_scale, workers)
    limits, cmaps = limits_from_products(channels, frames, products, largest_zoom)
    for zoom in args.fov:
        render_movie(channels, frames, output_path(args.output_prefix, zoom, args.format),
                     zoom, native_fov, args.pixel_scale, limits, cmaps,
                     products, args.fps, args.dpi, args.asinh_a, args.pore_field,
                     args.tick_step)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
