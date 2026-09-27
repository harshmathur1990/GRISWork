#!/usr/bin/env python3
"""Standalone exported GRIS view. No original data files are required.

Requires Python 3.10+ and:
    python -m pip install numpy astropy matplotlib PySide6-Essentials
Run this file with Python. Use --check to validate the embedded data without a GUI.
The layout and selected quantities are fixed. Use the timeline, playback,
pan/zoom, save-image toolbar and optional GRIS flicker to inspect the view.
"""
import base64
from functools import lru_cache
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class ExportBundle:
    def __init__(self, path):
        self.stream = tempfile.TemporaryFile()
        try:
            with open(path, 'rb') as source:
                for line in source:
                    if line.rstrip() == b'# GRIS_EMBEDDED_DATA_BEGIN':
                        break
                else:
                    raise ValueError('Embedded data is missing.')
                for line in source:
                    if line.rstrip() == b'# GRIS_EMBEDDED_DATA_END':
                        break
                    if not line.startswith(b'#|'):
                        raise ValueError('Invalid embedded data line.')
                    self.stream.write(base64.b85decode(line[2:].strip()))
                else:
                    raise ValueError('Embedded data is truncated.')
            self.stream.seek(0)
            self.archive = zipfile.ZipFile(self.stream)
            self.manifest = json.loads(self.archive.read('manifest.json'))
            if self.manifest['format_version'] != 1:
                raise ValueError('Unsupported export format.')
        except Exception:
            self.stream.close()
            raise

    @lru_cache(maxsize=4)
    def array(self, name):
        with self.archive.open(name) as stream:
            array = np.load(stream, allow_pickle=False)
        array.setflags(write=False)
        return array

    def check(self):
        for name in self.archive.namelist():
            if name.endswith('.npy'):
                self.array(name)
        for panel in self.manifest['panels']:
            if len(panel['frames']) != len(self.manifest['times']):
                raise ValueError('Panel timeline length disagrees with timestamps.')
            for frame in panel['frames']:
                if 'error' not in frame:
                    if self.array(frame['data']).ndim != 2:
                        raise ValueError('Expected a 2-D image.')
                    WCS(fits.Header.fromstring(frame['wcs'], sep='\n'))
        self.array.cache_clear()

    def close(self):
        self.array.cache_clear()
        self.archive.close()
        self.stream.close()


def world_readout(wcs, x, y, absolute):
    wx, wy = wcs.pixel_to_world_values(x, y)
    if absolute:
        wx = (wx + 180) % 360 - 180
    wx = (wx * u.Unit(wcs.world_axis_units[0])).to_value(u.arcsec)
    wy = (wy * u.Unit(wcs.world_axis_units[1])).to_value(u.arcsec)
    return f'X={wx:.2f} arcsec, Y={wy:.2f} arcsec'


class SharedPanel(QtWidgets.QGroupBox):
    def __init__(self, bundle, spec, number):
        super().__init__(f'Panel {number} — {spec["source"]}')
        self.bundle, self.spec = bundle, spec
        self.overlay_artist = None
        self.figure = Figure(figsize=(4, 3), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(260, 200)
        layout = QtWidgets.QVBoxLayout(self)
        self.flicker = QtWidgets.QPushButton('Flicker GRIS')
        self.flicker.setCheckable(True)
        self.flicker.setEnabled(any(frame.get('overlay') for frame in spec['frames']))
        self.flicker.toggled.connect(self.toggle_flicker)
        layout.addWidget(self.flicker)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.detail)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.tick)

    def toggle_flicker(self, checked):
        if checked and self.overlay_artist is not None:
            self.overlay_artist.set_visible(True)
            self.flicker.setText('Stop flicker')
            self.timer.start()
        else:
            self.stop()
        self.canvas.draw_idle()

    def stop(self):
        self.timer.stop()
        self.flicker.blockSignals(True)
        self.flicker.setChecked(False)
        self.flicker.blockSignals(False)
        self.flicker.setText('Flicker GRIS')
        if self.overlay_artist is not None:
            self.overlay_artist.set_visible(False)

    def tick(self):
        if self.overlay_artist is not None:
            self.overlay_artist.set_visible(not self.overlay_artist.get_visible())
            self.canvas.draw_idle()

    def render(self, index):
        phase = self.overlay_artist is not None and self.overlay_artist.get_visible()
        self.figure.clear()
        self.overlay_artist = None
        frame = self.spec['frames'][index]
        if 'error' in frame:
            self.stop()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.set_title('Frame unavailable')
            self.detail.setText(frame['error'])
            self.canvas.draw_idle()
            return
        wcs = WCS(fits.Header.fromstring(frame['wcs'], sep='\n'))
        self.ax = self.figure.add_subplot(111, projection=wcs)
        if frame['absolute']:
            self.ax.coords[0].set_coord_type('longitude', coord_wrap=180 * u.deg)
            self.ax.coords[1].set_coord_type('latitude')
        labels = ('Solar X [arcsec]', 'Solar Y [arcsec]') if frame['absolute'] else (
            'ΔX from GRIS centre [arcsec]', 'ΔY from GRIS centre [arcsec]')
        for i, label in enumerate(labels):
            self.ax.coords[i].set_format_unit(u.arcsec)
            self.ax.coords[i].set_major_formatter('s.s' if frame['absolute'] else 'x.x')
            self.ax.coords[i].set_axislabel(label)
            self.ax.coords[i].set_ticks(number=4)
            self.ax.coords[i].set_ticklabel(size=8, exclude_overlapping=True)
        self.ax.format_coord = lambda x, y: world_readout(wcs, x, y, frame['absolute'])
        data = self.bundle.array(frame['data'])
        self.artist = self.ax.imshow(data, origin='lower', interpolation='nearest',
                                    cmap=self.spec['cmap'], vmin=frame['limits'][0], vmax=frame['limits'][1])
        self.figure.colorbar(self.artist, ax=self.ax).set_label(frame['unit'])
        if frame.get('overlay'):
            self.overlay_artist = self.ax.imshow(
                np.ma.masked_invalid(self.bundle.array(frame['overlay'])), origin='lower',
                cmap='gray', vmin=frame['overlay_limits'][0], vmax=frame['overlay_limits'][1],
                extent=frame['overlay_extent'], visible=self.flicker.isChecked() and phase)
        if frame.get('boundary'):
            self.ax.plot(*frame['boundary'], color='cyan', linewidth=1.2)
        if self.overlay_artist is None:
            self.stop()
        self.ax.set_xlim(self.spec['xlim'])
        self.ax.set_ylim(self.spec['ylim'])
        self.ax.set_title(self.spec['title'], fontsize=10)
        self.detail.setText(frame['detail'])
        self.canvas.draw_idle()


class SharedViewer(QtWidgets.QWidget):
    def __init__(self, bundle):
        super().__init__()
        self.bundle = bundle
        manifest = bundle.manifest
        self.setWindowTitle('Shared GRIS view — fixed data selection')
        self.resize(*manifest['window_size'])
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(QtWidgets.QLabel('Shared view: data selections and grid are fixed; pan/zoom and playback are available.'))
        row = QtWidgets.QHBoxLayout()
        outer.addLayout(row)
        previous, following = QtWidgets.QPushButton('◀'), QtWidgets.QPushButton('▶')
        self.play = QtWidgets.QPushButton('Play')
        self.play.setCheckable(True)
        self.time = QtWidgets.QComboBox()
        self.time.addItems([f'{i+1}/{len(manifest["times"])}  {stamp} UTC'
                           for i, stamp in enumerate(manifest['times'])])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(manifest['times']) - 1)
        self.fps = QtWidgets.QDoubleSpinBox()
        self.fps.setRange(.1, 30)
        self.fps.setValue(manifest['fps'])
        self.fps.setSuffix(' fps')
        self.loop = QtWidgets.QCheckBox('Loop')
        self.loop.setChecked(manifest['loop'])
        for control in (previous, self.play, following, self.time, self.slider, self.fps, self.loop):
            row.addWidget(control)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.fps.valueChanged.connect(lambda: self.timer.setInterval(round(1000 / self.fps.value())))
        self.play.toggled.connect(self.toggle_play)
        previous.clicked.connect(lambda: self.set_time(max(0, self.index - 1)))
        following.clicked.connect(lambda: self.set_time(min(len(manifest['times']) - 1, self.index + 1)))
        self.time.currentIndexChanged.connect(self.set_time)
        self.slider.valueChanged.connect(self.set_time)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(host)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)
        self.panels = []
        for i, spec in enumerate(manifest['panels']):
            panel = SharedPanel(bundle, spec, i + 1)
            grid.addWidget(panel, spec['row'], spec['column'])
            self.panels.append(panel)
        self.set_time(manifest['current_time'])
        for panel in self.panels:
            if panel.spec['flicker']:
                panel.flicker.setChecked(True)

    def set_time(self, index):
        self.index = index
        for control in (self.time, self.slider):
            control.blockSignals(True)
        self.time.setCurrentIndex(index)
        self.slider.setValue(index)
        for control in (self.time, self.slider):
            control.blockSignals(False)
        for panel in self.panels:
            panel.render(index)

    def toggle_play(self, playing):
        self.play.setText('Pause' if playing else 'Play')
        if playing:
            if self.index == len(self.bundle.manifest['times']) - 1:
                self.set_time(0)
            self.timer.start(round(1000 / self.fps.value()))
        else:
            self.timer.stop()

    def advance(self):
        if self.index + 1 < len(self.bundle.manifest['times']):
            self.set_time(self.index + 1)
        elif self.loop.isChecked():
            self.set_time(0)
        else:
            self.play.setChecked(False)

    def closeEvent(self, event):
        self.timer.stop()
        for panel in self.panels:
            panel.stop()
        super().closeEvent(event)


def main():
    bundle = ExportBundle(__file__)
    try:
        if '--check' in sys.argv:
            bundle.check()
            print(f'Valid export: {len(bundle.manifest["panels"])} panels, '
                  f'{len(bundle.manifest["times"])} time frames.')
            return 0
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        viewer = SharedViewer(bundle)
        viewer.show()
        return app.exec()
    finally:
        bundle.close()


if __name__ == '__main__':
    sys.exit(main())
