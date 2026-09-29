#!/usr/bin/env python3
"""Make a two-row movie of the AIA observations matched to 30 GRIS scans.

The root ``alignment.json`` is authoritative: it supplies exactly one nearest
AIA observation for each GRIS timestamp.  Columns are the available AIA
channels.  The upper and lower rows show 4x and 2x the instantaneous GRIS
field of view, respectively.

Each AIA channel gets one normalization shared by both rows and every movie
frame.  Limits are scanned from the complete selected time series before
rendering, but colorbars are intentionally omitted.  By default the limits are
the true finite minimum and maximum, so no finite image value is
saturated/clipped by the color limits.

Example
-------
python make_aia_gris_fov_movie.py \
    --aligned-root /mn/stornext/d9/data/harshm/GRISData/aligned_SDO \
    --output aia_gris_fov.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_pixel, proj_plane_pixel_scales


GOOD_STATUSES = {"written", "existing"}


@dataclass(frozen=True)
class MovieFrame:
    index: int
    gris_time: str
    gris_header: Path
    aia_paths: dict[str, Path]
    time_offsets: dict[str, float]


@dataclass(frozen=True)
class Crop:
    data: np.ndarray
    gris_x: np.ndarray
    gris_y: np.ndarray
    extent_arcsec: tuple[float, float, float, float]


def _display_channel(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("AIA/") else name


def _canonical_channel(name: str) -> str:
    return name if name.startswith("AIA/") else f"AIA/{name}"


def load_frames(aligned_root: Path, requested_channels: Sequence[str] | None,
                expected_frames: int | None) -> tuple[list[str], list[MovieFrame]]:
    """Load and validate the manifest-selected AIA files and GRIS headers."""
    manifest_path = aligned_root / "alignment.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing alignment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest.get("frames", []), key=lambda row: int(row["frame_index"]))
    if not rows:
        raise ValueError("alignment.json contains no frames")
    indices = [int(row["frame_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("Manifest frame_index values must be consecutive from zero")
    if expected_frames is not None and len(rows) != expected_frames:
        raise ValueError(f"Expected {expected_frames} GRIS frames, found {len(rows)}")

    available = sorted({name for row in rows for name in row.get("channels", {})
                        if name.startswith("AIA/")}, key=_display_channel)
    if requested_channels:
        channels = list(dict.fromkeys(_canonical_channel(name) for name in requested_channels))
        missing = sorted(set(channels) - set(available))
        if missing:
            raise ValueError(f"Requested AIA channel(s) absent from alignment.json: {', '.join(missing)}")
    else:
        channels = available
    if not channels:
        raise ValueError("No AIA channels are available in alignment.json")

    result = []
    for row in rows:
        index = int(row["frame_index"])
        header_value = row.get("gris_header")
        header_path = (aligned_root / header_value if header_value else
                       aligned_root / "HMI/Continuum/gris_wcs" / f"gris_{index:04d}.hdr")
        if not header_path.is_file():
            raise FileNotFoundError(f"Missing GRIS WCS header for frame {index}: {header_path}")
        paths: dict[str, Path] = {}
        offsets: dict[str, float] = {}
        for channel in channels:
            record = row.get("channels", {}).get(channel)
            if not record or record.get("status") not in GOOD_STATUSES or not record.get("registered_sdo"):
                reason = record.get("reason", "no usable manifest association") if record else "missing record"
                raise ValueError(f"Frame {index}, {channel}: {reason}")
            path = aligned_root / record["registered_sdo"]
            if not path.is_file():
                raise FileNotFoundError(f"Frame {index}, {channel}: missing registered FITS: {path}")
            paths[channel] = path
            offsets[channel] = float(record.get("sdo_minus_gris_seconds", math.nan))
        result.append(MovieFrame(index, row["timestamp_utc"], header_path, paths, offsets))
    return channels, result


def read_gris_wcs(path: Path) -> tuple[WCS, tuple[int, int]]:
    header = fits.Header.fromtextfile(path)
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    wcs = WCS(header)
    if min(shape) <= 0 or not wcs.has_celestial:
        raise ValueError(f"Invalid GRIS WCS header: {path}")
    return wcs, shape


def crop_around_gris(data: np.ndarray, aia_wcs: WCS, gris_wcs: WCS,
                     gris_shape: tuple[int, int], zoom: float) -> Crop:
    """Return an AIA pixel crop centred on a scaled GRIS WCS footprint."""
    if data.ndim != 2:
        raise ValueError(f"AIA data must be two-dimensional, got {data.shape}")
    if not math.isfinite(zoom) or zoom <= 0:
        raise ValueError("FOV multiplier must be finite and positive")
    gris_ny, gris_nx = gris_shape
    corner_x = np.array([-.5, gris_nx - .5, gris_nx - .5, -.5, -.5])
    corner_y = np.array([-.5, -.5, gris_ny - .5, gris_ny - .5, -.5])
    aia_x, aia_y = pixel_to_pixel(gris_wcs, aia_wcs, corner_x, corner_y)
    if not np.all(np.isfinite([aia_x, aia_y])):
        raise ValueError("GRIS footprint does not transform to finite AIA pixels")

    left0, right0 = float(np.min(aia_x)), float(np.max(aia_x))
    bottom0, top0 = float(np.min(aia_y)), float(np.max(aia_y))
    cx, cy = (left0 + right0) / 2, (bottom0 + top0) / 2
    half_width = max((right0 - left0) * zoom / 2, .5)
    half_height = max((top0 - bottom0) * zoom / 2, .5)
    ny, nx = data.shape
    x0 = max(0, int(math.floor(cx - half_width)))
    x1 = min(nx, int(math.ceil(cx + half_width)) + 1)
    y0 = max(0, int(math.floor(cy - half_height)))
    y1 = min(ny, int(math.ceil(cy + half_height)) + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Scaled GRIS field of view lies outside the AIA image")
    # Express the displayed axes as offsets from the centre of the GRIS
    # footprint.  Registered AIA data have nearly orthogonal image axes; the
    # projected pixel scales retain the correct angular size even if the crop
    # dimensions vary slightly between frames.
    scales = np.asarray(proj_plane_pixel_scales(aia_wcs.celestial), dtype=float) * 3600.0
    if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Cannot determine finite AIA pixel scales in arcseconds")
    local_cx, local_cy = cx - x0, cy - y0
    extent = ((-.5 - local_cx) * scales[0], (x1 - x0 - .5 - local_cx) * scales[0],
              (-.5 - local_cy) * scales[1], (y1 - y0 - .5 - local_cy) * scales[1])
    return Crop(np.asarray(data[y0:y1, x0:x1], dtype=float),
                (aia_x - x0 - local_cx) * scales[0],
                (aia_y - y0 - local_cy) * scales[1], extent)


def scan_limits(channels: Sequence[str], frames: Sequence[MovieFrame], zoom: float
                ) -> tuple[dict[str, tuple[float, float]], dict[str, object], dict[str, str]]:
    """Find exact finite limits over all selected frames, one pair per channel."""
    import sunpy.map

    limits = {channel: [math.inf, -math.inf] for channel in channels}
    cmaps: dict[str, object] = {}
    units: dict[str, str] = {}
    for number, frame in enumerate(frames, 1):
        gris_wcs, gris_shape = read_gris_wcs(frame.gris_header)
        for channel in channels:
            aia = sunpy.map.Map(str(frame.aia_paths[channel]))
            crop = crop_around_gris(aia.data, aia.wcs, gris_wcs, gris_shape, zoom).data
            finite = crop[np.isfinite(crop)]
            if finite.size:
                limits[channel][0] = min(limits[channel][0], float(np.min(finite)))
                limits[channel][1] = max(limits[channel][1], float(np.max(finite)))
            cmaps.setdefault(channel, aia.plot_settings.get("cmap", "gray"))
            units.setdefault(channel, str(aia.unit or aia.meta.get("bunit", "native units")))
        print(f"Scanning fixed color limits: {number}/{len(frames)}", end="\r", flush=True)
    print()

    final: dict[str, tuple[float, float]] = {}
    for channel, (low, high) in limits.items():
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"No finite pixels in the {zoom:g}x crops for {channel}")
        if low == high:
            pad = max(abs(low) * 1e-6, 1e-12)
            low, high = low - pad, high + pad
        final[channel] = (low, high)
        print(f"{channel}: fixed limits across all frames = [{low:.8g}, {high:.8g}] {units[channel]}")
    return final, cmaps, units


def render_movie(channels: Sequence[str], frames: Sequence[MovieFrame], output: Path,
                 fps: float, dpi: int, large_zoom: float, small_zoom: float,
                 asinh_a: float) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import sunpy.map
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    # The larger crop contains the smaller one, so scanning it guarantees that
    # neither row can exceed the shared channel limits.
    limits, cmaps, units = scan_limits(channels, frames, max(large_zoom, small_zoom))
    norms = {channel: ImageNormalize(vmin=limits[channel][0], vmax=limits[channel][1],
                                     stretch=AsinhStretch(asinh_a), clip=True)
             for channel in channels}

    ncols = len(channels)
    fig, axes = plt.subplots(2, ncols, figsize=(5.0 * ncols, 8.2), squeeze=False,
                             constrained_layout=True)
    artists: dict[tuple[int, str], object] = {}
    boundaries: dict[tuple[int, str], object] = {}
    cached_crops: dict[tuple[int, str], Crop] = {}
    cached_frame_number: int | None = None

    def crops_for(frame_number: int) -> dict[tuple[int, str], Crop]:
        nonlocal cached_crops, cached_frame_number
        frame = frames[frame_number]
        gris_wcs, gris_shape = read_gris_wcs(frame.gris_header)
        result = {}
        for channel in channels:
            aia = sunpy.map.Map(str(frame.aia_paths[channel]))
            for row, zoom in enumerate((large_zoom, small_zoom)):
                result[row, channel] = crop_around_gris(
                    aia.data, aia.wcs, gris_wcs, gris_shape, zoom)
        cached_crops = result
        cached_frame_number = frame_number
        return result

    initial = crops_for(0)
    for column, channel in enumerate(channels):
        for row, zoom in enumerate((large_zoom, small_zoom)):
            ax = axes[row, column]
            crop = initial[row, channel]
            image = ax.imshow(crop.data, origin="lower", cmap=cmaps[channel],
                              norm=norms[channel], interpolation="nearest",
                              extent=crop.extent_arcsec)
            # Draw the exact transformed GRIS WCS footprint, not merely a
            # centre marker or an axis-aligned approximation.  The black
            # stroke keeps the cyan contour visible on both dark and bright
            # AIA structures.
            boundary, = ax.plot(crop.gris_x, crop.gris_y, color="cyan", lw=2.0,
                                alpha=1.0, label="GRIS FOV", zorder=5)
            boundary.set_path_effects([
                path_effects.Stroke(linewidth=3.5, foreground="black"),
                path_effects.Normal(),
            ])
            # One panel is sufficient to communicate the angular dimensions.
            # Keep every other image clean and place exactly two ticks on each
            # direction of the bottom-left panel.
            if row == 1 and column == 0:
                left, right, bottom, top = crop.extent_arcsec
                ax.set_xticks(np.linspace(left, right, 4)[1:3])
                ax.set_yticks(np.linspace(bottom, top, 4)[1:3])
                ax.set_xlabel("ΔX [arcsec]")
                ax.set_ylabel("ΔY [arcsec]")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(f"{zoom:g}× GRIS FOV" if column == 0 else "")
            if row == 0:
                ax.set_title(f"AIA {_display_channel(channel)} Å")
                ax.legend(loc="upper right", framealpha=.75, fontsize=8)
            artists[row, channel] = image
            boundaries[row, channel] = boundary
    title = fig.suptitle("")

    def update(frame_number: int):
        frame = frames[frame_number]
        crops = cached_crops if frame_number == cached_frame_number else crops_for(frame_number)
        for channel in channels:
            for row in range(2):
                crop = crops[row, channel]
                image = artists[row, channel]
                image.set_data(crop.data)
                image.set_extent(crop.extent_arcsec)
                ax = axes[row, channels.index(channel)]
                left, right, bottom, top = crop.extent_arcsec
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
                if row == 1 and channels.index(channel) == 0:
                    ax.set_xticks(np.linspace(left, right, 4)[1:3])
                    ax.set_yticks(np.linspace(bottom, top, 4)[1:3])
                boundaries[row, channel].set_data(crop.gris_x, crop.gris_y)
        offsets = ", ".join(
            f"{_display_channel(channel)}: Δt={frame.time_offsets[channel]:+.2f}s"
            if math.isfinite(frame.time_offsets[channel]) else _display_channel(channel)
            for channel in channels)
        title.set_text(f"GRIS {frame.index + 1:02d}/{len(frames)}  {frame.gris_time}  |  {offsets}")
        print(f"Rendering movie: {frame_number + 1}/{len(frames)}", end="\r", flush=True)
        return [*artists.values(), *boundaries.values(), title]

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False, repeat=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for MP4 output; install it or choose an .gif output")
        writer = FFMpegWriter(fps=fps, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
    else:
        raise ValueError("Output filename must end in .mp4 or .gif")
    animation.save(output, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"\nWrote {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path,
                        default=Path("/mn/stornext/d9/data/harshm/GRISData/aligned_SDO"),
                        help="Root containing alignment.json, AIA/, and HMI/Continuum/gris_wcs/")
    parser.add_argument("--output", type=Path, default=Path("aia_gris_fov.mp4"),
                        help="Output .mp4 or .gif (default: aia_gris_fov.mp4)")
    parser.add_argument("--channels", nargs="+", metavar="WAVELENGTH",
                        help="Optional AIA channels, e.g. 171 304; default: all AIA channels in the manifest")
    parser.add_argument("--expected-frames", type=int, default=30,
                        help="Require this many GRIS frames (default: 30; use 0 to accept any count)")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--top-fov", type=float, default=4.0,
                        help="Top-row GRIS FOV multiplier (default: 4)")
    parser.add_argument("--bottom-fov", type=float, default=2.0,
                        help="Bottom-row GRIS FOV multiplier (default: 2)")
    parser.add_argument("--asinh-a", type=float, default=0.01,
                        help="Asinh stretch transition parameter (default: 0.01; limits remain exact)")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_frames < 0:
        raise ValueError("--expected-frames must be non-negative")
    for option, value in (("--fps", args.fps), ("--top-fov", args.top_fov),
                          ("--bottom-fov", args.bottom_fov), ("--asinh-a", args.asinh_a)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{option} must be finite and positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    expected = args.expected_frames or None
    channels, frames = load_frames(args.aligned_root, args.channels, expected)
    print(f"Using {len(frames)} GRIS timestamps and AIA channels: {', '.join(channels)}")
    render_movie(channels, frames, args.output, args.fps, args.dpi,
                 args.top_fov, args.bottom_fov, args.asinh_a)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
