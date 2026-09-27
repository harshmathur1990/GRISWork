"""Write selected panel slices and a standalone player into one Python file."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from astropy.wcs.utils import pixel_to_pixel


class ExportCancelled(Exception):
    pass


def snapshot_view(viewer):
    """Capture the applied grid, selections and appearance before reading data."""
    panels = []
    for panel in viewer.panels:
        row, column, _, _ = viewer.grid.getItemPosition(viewer.grid.indexOf(panel))
        mode = panel.scaling.currentIndex()
        limits = panel.limits
        if mode == 2:
            limits = (float(panel.low.text()), float(panel.high.text()))
            if not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Fix invalid manual colour limits before exporting.')
        if mode == 1 and limits is None:
            raise ValueError('Render the held colour limits before exporting.')
        source = panel.source.currentText()
        if source == 'Atmosphere':
            title = f'{panel.parameter.currentText()} | log τ ≈ {panel.coordinate.value():.3f}'
        elif source.startswith('SDO:'):
            title = source + ' | Full disk'
        else:
            title = f'{source} {panel.stokes.currentText()} | λ {panel.coordinate.value():.4f} Å'
        panels.append(dict(
            row=row, column=column, source=source,
            index=panel.index.value(), stokes=panel.stokes.currentIndex(),
            parameter=panel.parameter.currentText(), coordinate=panel.coordinate.value(),
            cmap=panel.cmap.currentText(), scaling=mode,
            fixed_limits=list(map(float, limits)) if limits is not None else None,
            xlim=list(map(float, panel.ax.get_xlim())), ylim=list(map(float, panel.ax.get_ylim())),
            title=title, flicker=panel.flicker.isChecked()))
    return dict(format_version=1, times=[stamp.isoformat() for stamp in viewer.store.times],
                current_time=viewer.time_index, fps=viewer.fps.value(), loop=viewer.loop.isChecked(),
                window_size=[max(800, viewer.width()), max(600, viewer.height())],
                tolerance=viewer.tolerance.value(), panels=panels)


def image_limits(data):
    values = data[np.isfinite(data)]
    if not values.size:
        raise ValueError('This slice has no finite data.')
    lo, hi = np.percentile(values, [1, 99])
    if lo == hi:
        lo, hi = lo - .5, hi + .5
    return [float(lo), float(hi)]


def export_view(store, snapshot, destination, progress=None):
    """Lossless, selected-slice export. No original file paths are needed to play it.

    progress(done, total, message) may return False to cancel. A temporary archive
    and output file bound memory use; destination is replaced only on success.
    """
    destination = Path(destination)
    manifest = json.loads(json.dumps(snapshot))
    total = len(manifest['panels']) * len(manifest['times'])
    done = unavailable = 0

    def notify(message):
        if progress is not None and progress(done, total, message) is False:
            raise ExportCancelled('Export cancelled; no output file was replaced.')

    notify('Preparing selected data…')
    with tempfile.TemporaryDirectory(prefix='gris-export-') as temporary:
        archive_path = Path(temporary) / 'data.zip'
        arrays = set()
        with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as archive:
            def add_array(values):
                array = np.ascontiguousarray(values)
                if array.dtype.hasobject:
                    raise ValueError('Only numeric image arrays can be exported.')
                digest = hashlib.sha256(array.dtype.str.encode() + repr(array.shape).encode())
                digest.update(memoryview(array).cast('B'))
                name = f'arrays/{digest.hexdigest()}.npy'
                if name not in arrays:
                    with archive.open(name, 'w', force_zip64=True) as stream:
                        np.lib.format.write_array(stream, array, allow_pickle=False)
                    arrays.add(name)
                return name

            for number, spec in enumerate(manifest['panels'], 1):
                frames = []
                for time_index in range(len(manifest['times'])):
                    notify(f'Panel {number}/{len(manifest["panels"])}, time {time_index + 1}/{len(manifest["times"])}')
                    try:
                        view = store.display(spec['source'], time_index, spec['index'], spec['stokes'],
                                             spec['parameter'], manifest['tolerance'])
                        auto_limits = image_limits(view.data)
                    except (ValueError, OSError, KeyError, RuntimeError) as error:
                        frames.append(dict(error=str(error)))
                        unavailable += 1
                        done += 1
                        continue
                    record = dict(data=add_array(view.data), unit=view.unit, detail=view.detail,
                                  wcs=view.wcs.to_header(relax=True).tostring(sep='\n', padding=False),
                                  absolute=view.absolute,
                                  limits=auto_limits if spec['scaling'] == 0 else spec['fixed_limits'])
                    if view.gris_data is not None:
                        ny, nx = view.gris_data.shape
                        bx, by = pixel_to_pixel(view.gris_wcs, view.wcs,
                                                np.array([-.5, nx-.5, nx-.5, -.5, -.5]),
                                                np.array([-.5, -.5, ny-.5, ny-.5, -.5]))
                        record['boundary'] = [bx.tolist(), by.tolist()]
                        if view.overlay is not None and np.isfinite(view.gris_data).any():
                            record.update(overlay=add_array(view.overlay),
                                          overlay_extent=list(map(float, view.overlay_extent)),
                                          overlay_limits=image_limits(view.gris_data))
                    frames.append(record)
                    done += 1
                spec['frames'] = frames
            manifest['unavailable_frames'] = unavailable
            archive.writestr('manifest.json', json.dumps(manifest))
        notify('Writing self-contained Python file…')
        runtime = Path(__file__).with_name('inversion_viewer_export_runtime.py').read_bytes()
        pending = None
        try:
            with tempfile.NamedTemporaryFile('wb', dir=destination.parent, prefix='.gris-export-',
                                             suffix='.tmp', delete=False) as output:
                pending = Path(output.name)
                output.write(runtime)
                output.write(b'\n# GRIS_EMBEDDED_DATA_BEGIN\n')
                with archive_path.open('rb') as source:
                    blocks = 0
                    while block := source.read(4096):
                        output.write(b'#|' + base64.b85encode(block) + b'\n')
                        blocks += 1
                        if blocks % 256 == 0:
                            notify('Writing self-contained Python file…')
                output.write(b'# GRIS_EMBEDDED_DATA_END\n')
            notify('Finishing export…')
            os.replace(pending, destination)
            pending = None
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    return dict(bytes=destination.stat().st_size, arrays=len(arrays), unavailable=unavailable,
                panels=len(manifest['panels']), times=len(manifest['times']))
