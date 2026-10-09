#!/usr/bin/env python3
"""Make separate 4x and 2x GRIS-FOV movies at the original AIA cadence.

The original AIA FITS sequences under ``<raw-root>/AIA`` drive the movie. The
shortest-cadence AIA channel is the master clock; every other AIA channel and
HMI magnetogram is matched to its nearest exposure. The GRIS WCS header nearest
each AIA time supplies the field-of-view outline and crop centre.

Each output is a fixed 3x2 layout: up to five AIA channels followed by HMI
magnetogram as panel six. Exactly one panel has two fixed ticks per direction;
the angular limits never change, so labels cannot make panels jump. Red and
blue contours show positive and negative HMI field on every panel. No colorbars
are drawn. AIA limits are fixed per channel over the full time series, using
true finite extrema so brightenings remain comparable and are not clipped.

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
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel

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
    gris: GrisFrame
    aia: dict[str, TimedFile]
    magnetogram: TimedFile


@dataclass(frozen=True)
class SampledView:
    data: np.ndarray
    source_x: np.ndarray
    source_y: np.ndarray
    source_wcs: WCS
    gris_x: np.ndarray
    gris_y: np.ndarray
    extent: tuple[float, float, float, float]


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
                       indexed: dict[str, list[TimedFile]], magnetograms: Sequence[TimedFile],
                       gris_frames: Sequence[GrisFrame]) -> list[MovieFrame]:
    frames = []
    for index, clock in enumerate(indexed[master]):
        gris = min(gris_frames, key=lambda item: (abs(item.time - clock.time), item.time, item.index))
        frames.append(MovieFrame(index, clock.time, gris,
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


def gris_dimensions(gris_frames: Sequence[GrisFrame]) -> tuple[float, float]:
    """Maximum axis-aligned GRIS footprint dimensions in arcseconds."""
    widths, heights = [], []
    for frame in gris_frames:
        wcs, (ny, nx) = read_gris_wcs(frame.header)
        matrix = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=float) * 3600.0
        offsets = np.array([[-nx / 2, -ny / 2], [nx / 2, -ny / 2],
                            [nx / 2, ny / 2], [-nx / 2, ny / 2]]) @ matrix.T
        widths.append(float(np.ptp(offsets[:, 0])))
        heights.append(float(np.ptp(offsets[:, 1])))
    if not np.all(np.isfinite(widths + heights)) or min(widths + heights) <= 0:
        raise ValueError("Cannot determine finite GRIS angular dimensions")
    return max(widths), max(heights)


def load_map(path: Path, register_aia: bool):
    import sunpy.map

    result = sunpy.map.Map(str(path))
    if register_aia:
        from aiapy.calibrate import register
        result = register(result)
    if result.data.ndim != 2:
        raise ValueError(f"Expected a 2-D image: {path}")
    return result


def sample_view(source, gris_wcs: WCS, gris_shape: tuple[int, int],
                width_arcsec: float, height_arcsec: float) -> SampledView:
    """Sample a fixed-size view centred on the current GRIS footprint."""
    from astropy.wcs.utils import proj_plane_pixel_scales
    from scipy.ndimage import map_coordinates

    gris_ny, gris_nx = gris_shape
    corner_x = np.array([-.5, gris_nx - .5, gris_nx - .5, -.5, -.5])
    corner_y = np.array([-.5, -.5, gris_ny - .5, gris_ny - .5, -.5])
    source_x, source_y = pixel_to_pixel(gris_wcs, source.wcs, corner_x, corner_y)
    centre_x, centre_y = pixel_to_pixel(
        gris_wcs, source.wcs, (gris_nx - 1) / 2, (gris_ny - 1) / 2)
    if (not np.all(np.isfinite(source_x)) or not np.all(np.isfinite(source_y))
            or not np.isfinite(centre_x) or not np.isfinite(centre_y)):
        raise ValueError("GRIS footprint does not transform to finite source pixels")

    scales = np.asarray(proj_plane_pixel_scales(source.wcs.celestial), dtype=float) * 3600.0
    if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Cannot determine finite source pixel scales")
    nx = max(2, int(math.ceil(width_arcsec / scales[0])))
    ny = max(2, int(math.ceil(height_arcsec / scales[1])))
    dx, dy = width_arcsec / nx, height_arcsec / ny
    x_offsets = -width_arcsec / 2 + (np.arange(nx) + .5) * dx
    y_offsets = -height_arcsec / 2 + (np.arange(ny) + .5) * dy
    grid_x, grid_y = np.meshgrid(centre_x + x_offsets / scales[0],
                                 centre_y + y_offsets / scales[1])
    data = map_coordinates(np.asarray(source.data, dtype=float), [grid_y, grid_x],
                           order=1, mode="constant", cval=np.nan)
    extent = (-width_arcsec / 2, width_arcsec / 2,
              -height_arcsec / 2, height_arcsec / 2)
    return SampledView(data, grid_x, grid_y, source.wcs,
                       (source_x - centre_x) * scales[0],
                       (source_y - centre_y) * scales[1], extent)


def sample_magnetogram(magnetogram, view: SampledView) -> np.ndarray:
    """Sample HMI field onto a displayed panel's source-coordinate grid."""
    from scipy.ndimage import map_coordinates

    mx, my = pixel_to_pixel(view.source_wcs, magnetogram.wcs,
                            view.source_x, view.source_y)
    return map_coordinates(np.asarray(magnetogram.data, dtype=float), [my, mx],
                           order=1, mode="constant", cval=np.nan)


def scan_limits(channels: Sequence[str], frames: Sequence[MovieFrame],
                width: float, height: float, register_aia: bool
                ) -> tuple[dict[str, tuple[float, float]], dict[str, object]]:
    """Compute fixed full-series limits once, including the HMI sixth panel."""
    limits = {channel: [math.inf, -math.inf] for channel in channels}
    hmi_absmax = 0.0
    cmaps: dict[str, object] = {}
    for number, frame in enumerate(frames, 1):
        gris_wcs, gris_shape = read_gris_wcs(frame.gris.header)
        for channel in channels:
            source = load_map(frame.aia[channel].path, register_aia)
            values = sample_view(source, gris_wcs, gris_shape, width, height).data
            finite = values[np.isfinite(values)]
            if finite.size:
                limits[channel][0] = min(limits[channel][0], float(np.min(finite)))
                limits[channel][1] = max(limits[channel][1], float(np.max(finite)))
            cmaps.setdefault(channel, source.plot_settings.get("cmap", "gray"))
        magnetogram = load_map(frame.magnetogram.path, False)
        values = sample_view(magnetogram, gris_wcs, gris_shape, width, height).data
        finite = values[np.isfinite(values)]
        if finite.size:
            hmi_absmax = max(hmi_absmax, float(np.max(np.abs(finite))))
        print(f"Scanning fixed display limits: {number}/{len(frames)}", end="\r", flush=True)
    print()

    final: dict[str, tuple[float, float]] = {}
    for channel, (low, high) in limits.items():
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"No finite pixels found for {channel}")
        if low == high:
            pad = max(abs(low) * 1e-6, 1e-12)
            low, high = low - pad, high + pad
        final[channel] = (low, high)
        print(f"{channel}: fixed full-series limits [{low:.8g}, {high:.8g}]")
    if not math.isfinite(hmi_absmax) or hmi_absmax <= 0:
        raise ValueError("No finite non-zero HMI magnetogram pixels found")
    final[HMI_CHANNEL] = (-hmi_absmax, hmi_absmax)
    cmaps[HMI_CHANNEL] = "gray"
    print(f"{HMI_CHANNEL}: fixed symmetric limits [{-hmi_absmax:.8g}, {hmi_absmax:.8g}]")
    return final, cmaps


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
                 zoom: float, gris_size: tuple[float, float], limits: dict[str, tuple[float, float]],
                 cmaps: dict[str, object], fps: float, dpi: int, asinh_a: float,
                 pore_field: float, register_aia: bool) -> None:
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    width, height = gris_size[0] * zoom, gris_size[1] * zoom
    panels = list(channels) + [None] * (5 - len(channels)) + [HMI_CHANNEL]
    tick_panel = next((index for index in range(3, 6) if panels[index] is not None), 0)
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
        gris_wcs, gris_shape = read_gris_wcs(frame.gris.header)
        magnetogram = load_map(frame.magnetogram.path, False)
        result: dict[int, tuple[SampledView, np.ndarray]] = {}
        for panel_index, channel in enumerate(panels):
            if channel is None:
                continue
            source = magnetogram if channel == HMI_CHANNEL else load_map(
                frame.aia[channel].path, register_aia)
            view = sample_view(source, gris_wcs, gris_shape, width, height)
            field = view.data if channel == HMI_CHANNEL else sample_magnetogram(magnetogram, view)
            result[panel_index] = view, field
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
        if panel_index == tick_panel:
            ax.set_xticks([-width / 4, width / 4])
            ax.set_yticks([-height / 4, height / 4])
            ax.set_xlabel("ΔX [arcsec]")
            ax.set_ylabel("ΔY [arcsec]")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
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
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
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
    parser.add_argument("--pore-field", type=float, default=500.0, metavar="GAUSS",
                        help="Absolute HMI levels for negative/positive pore contours (default: 500 G)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--asinh-a", type=float, default=0.01,
                        help="AIA asinh stretch transition parameter (default: 0.01)")
    parser.add_argument("--no-register-aia", action="store_true",
                        help="Do not apply aiapy in-memory registration to original level-1 AIA files")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for option, value in (("--fps", args.fps), ("--pore-field", args.pore_field),
                          ("--asinh-a", args.asinh_a)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{option} must be finite and positive")
    if args.dpi <= 0 or not args.fov or any(not math.isfinite(v) or v <= 0 for v in args.fov):
        raise ValueError("--dpi and every --fov value must be positive")

    manifest, gris_frames = load_manifest(args.aligned_root)
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
    frames = build_movie_frames(master, channels, indexed, magnetograms, gris_frames)
    if len(channels) < 5:
        print(f"Warning: found {len(channels)} AIA channels; unused AIA slots will be blank.",
              file=sys.stderr)
    print(f"Master AIA cadence: {master} ({len(frames)} original exposures); "
          f"panels: {', '.join(channels)}, {HMI_CHANNEL}")

    gris_size = gris_dimensions(gris_frames)
    largest_zoom = max(args.fov)
    limits, cmaps = scan_limits(channels, frames, gris_size[0] * largest_zoom,
                                gris_size[1] * largest_zoom, not args.no_register_aia)
    for zoom in args.fov:
        render_movie(channels, frames, output_path(args.output_prefix, zoom, args.format),
                     zoom, gris_size, limits, cmaps, args.fps, args.dpi, args.asinh_a,
                     args.pore_field, not args.no_register_aia)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
