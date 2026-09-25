#!/usr/bin/env python3
"""Configurable Qt comparison widget for GRIS observations, inversions and SDO."""
import argparse
from pathlib import Path
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from inversion_viewer_data import DataStore


class ImagePanel(QtWidgets.QGroupBox):
    def __init__(self, viewer, number):
        super().__init__(f'Panel {number}')
        self.viewer = viewer
        self.store = viewer.store
        self.artist = self.colorbar = None
        self.limits = None
        layout = QtWidgets.QVBoxLayout(self)
        self.source = QtWidgets.QComboBox()
        self.source.addItems(self.store.sources)
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.source, 1)
        settings = QtWidgets.QToolButton()
        settings.setText('Settings')
        settings.setCheckable(True)
        source_row.addWidget(settings)
        layout.addLayout(source_row)
        settings_host = QtWidgets.QWidget()
        controls = QtWidgets.QGridLayout(settings_host)
        controls.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(settings_host)
        settings_host.setVisible(False)
        settings.toggled.connect(settings_host.setVisible)
        self.parameter = QtWidgets.QComboBox()
        self.parameter.addItems(self.store.parameters)
        self.stokes = QtWidgets.QComboBox()
        self.index = QtWidgets.QSpinBox()
        self.coordinate = QtWidgets.QDoubleSpinBox()
        self.coordinate.setDecimals(6)
        self.coordinate.setRange(-1e9, 1e9)
        self.coordinate.setKeyboardTracking(False)
        self.index.setKeyboardTracking(False)
        self.index_label = QtWidgets.QLabel('Wavelength index (0-based)')
        self.coordinate_label = QtWidgets.QLabel('λ [Å], nearest sample')
        for row, (label, control) in enumerate([
                ('Atmosphere field', self.parameter),
                ('Stokes', self.stokes), (self.index_label, self.index),
                (self.coordinate_label, self.coordinate)]):
            controls.addWidget(QtWidgets.QLabel(label) if isinstance(label, str) else label, row, 0)
            controls.addWidget(control, row, 1)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(['gray', 'viridis', 'magma', 'inferno', 'RdBu_r', 'coolwarm'])
        self.scaling = QtWidgets.QComboBox()
        self.scaling.addItems(['Auto each frame (1–99%)', 'Hold limits', 'Manual limits'])
        self.low = QtWidgets.QLineEdit()
        self.high = QtWidgets.QLineEdit()
        self.low.setPlaceholderText('min')
        self.high.setPlaceholderText('max')
        controls.addWidget(self.cmap, 4, 0)
        controls.addWidget(self.scaling, 4, 1)
        controls.addWidget(self.low, 5, 0)
        controls.addWidget(self.high, 5, 1)
        self.figure = Figure(figsize=(4, 3), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumSize(260, 200)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.detail)
        self.source.currentTextChanged.connect(self.configure)
        self.parameter.currentTextChanged.connect(self.reset_limits)
        self.stokes.currentIndexChanged.connect(self.reset_limits)
        self.index.valueChanged.connect(self.index_changed)
        self.coordinate.editingFinished.connect(self.coordinate_changed)
        self.cmap.currentTextChanged.connect(self.render)
        self.scaling.currentTextChanged.connect(self.render)
        self.low.editingFinished.connect(self.render)
        self.high.editingFinished.connect(self.render)
        self.configure()

    def configure(self, *_):
        source = self.source.currentText()
        spectral = source in ('Observation', 'Fitted profiles')
        atmosphere = source == 'Atmosphere'
        self.parameter.setEnabled(atmosphere)
        self.stokes.blockSignals(True)
        self.stokes.clear()
        count = (self.store.ns if source == 'Observation' else
                 self.store.fit['profiles'].shape[-1] if source == 'Fitted profiles' else 1)
        self.stokes.addItems(list('IQUV')[:count])
        self.stokes.blockSignals(False)
        self.stokes.setEnabled(spectral)
        self.index.setEnabled(spectral or atmosphere)
        self.coordinate.setEnabled(spectral or atmosphere)
        self.index_label.setText('Depth index (0-based)' if atmosphere else 'Wavelength index (0-based)')
        self.coordinate_label.setText('log τ₅₀₀ at [t=0,y=0,x=0]' if atmosphere else 'λ [Å], nearest sample')
        self.coordinate.setToolTip('Depth indices select the same layer at every pixel. The displayed log τ is from the first pixel.')
        coords = self.store.coordinates(source)
        self.index.blockSignals(True)
        self.index.setRange(0, len(coords) - 1)
        target = -1 if atmosphere else 8542.09
        self.index.setValue(int(np.argmin(abs(coords - target))))
        self.index.blockSignals(False)
        self.index_changed()

    def index_changed(self, *_):
        self.coordinate.setValue(float(self.store.coordinates(self.source.currentText())[self.index.value()]))
        self.reset_limits()

    def coordinate_changed(self):
        coords = self.store.coordinates(self.source.currentText())
        self.index.setValue(int(np.argmin(abs(coords - self.coordinate.value()))))
        self.coordinate.setValue(float(coords[self.index.value()]))

    def reset_limits(self, *_):
        self.limits = None
        self.render()

    def render(self, *_):
        source = self.source.currentText()
        try:
            data, unit, detail = self.store.image(
                source, self.viewer.time_index, self.index.value(), self.stokes.currentIndex(),
                self.parameter.currentText(), self.viewer.time_mode.currentData(),
                self.viewer.tolerance.value())
            finite = data[np.isfinite(data)]
            if not finite.size:
                raise ValueError('This slice has no finite pixels.')
            mode = self.scaling.currentIndex()
            self.low.setEnabled(mode == 2)
            self.high.setEnabled(mode == 2)
            if mode == 2:
                limits = float(self.low.text()), float(self.high.text())
                if not all(np.isfinite(limits)) or limits[0] >= limits[1]:
                    raise ValueError('Manual minimum must be finite and less than maximum.')
            elif mode == 0 or self.limits is None:
                limits = tuple(np.percentile(finite, [1, 99]))
                if limits[0] == limits[1]:
                    limits = (limits[0] - .5, limits[1] + .5)
            else:
                limits = self.limits
            self.limits = limits
            if mode != 2:
                self.low.setText(f'{limits[0]:.6g}')
                self.high.setText(f'{limits[1]:.6g}')
            if self.artist is None:
                self.artist = self.ax.imshow(data, origin='lower', interpolation='nearest',
                                             cmap=self.cmap.currentText(), vmin=limits[0], vmax=limits[1])
                self.colorbar = self.figure.colorbar(self.artist, ax=self.ax)
                self.ax.set_xlabel('x [pixel]')
                self.ax.set_ylabel('y [pixel]')
            else:
                self.artist.set_data(data)
                self.artist.set_visible(True)
                self.artist.set_cmap(self.cmap.currentText())
                self.artist.set_clim(*limits)
                self.colorbar.ax.set_visible(True)
            self.colorbar.set_label(unit)
            if source == 'Atmosphere':
                title = f'{self.parameter.currentText()} | depth {self.index.value()} | log τ ≈ {self.coordinate.value():.3f}'
            elif source.startswith('SDO:'):
                title = source
            else:
                title = f'{source} {self.stokes.currentText()} | λ {self.coordinate.value():.4f} Å'
            self.ax.set_title(title, fontsize=10)
            self.detail.setText(detail)
            self.detail.setStyleSheet('')
        except Exception as exc:
            # Never leave a previous frame visible under a new timestamp on error.
            if self.artist is not None:
                self.artist.set_visible(False)
                self.colorbar.ax.set_visible(False)
            self.ax.set_title('Frame unavailable', fontsize=10)
            self.detail.setText(str(exc))
            self.detail.setStyleSheet('color: #b04030')
        self.canvas.draw_idle()


class ComparisonWidget(QtWidgets.QWidget):
    """Embeddable widget. The caller owns and closes the supplied DataStore."""
    def __init__(self, store, rows=2, columns=2, parent=None):
        super().__init__(parent)
        self.store = store
        self.time_index = 0
        self.panels = []
        outer = QtWidgets.QVBoxLayout(self)
        grid_bar = QtWidgets.QHBoxLayout()
        outer.addLayout(grid_bar)
        self.rows, self.columns = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        for label, control, value in [('Rows', self.rows, rows), ('Columns', self.columns, columns)]:
            control.setRange(1, 6)
            control.setValue(value)
            grid_bar.addWidget(QtWidgets.QLabel(label))
            grid_bar.addWidget(control)
        apply = QtWidgets.QPushButton('Apply grid')
        apply.clicked.connect(self.rebuild_grid)
        grid_bar.addWidget(apply)
        self.time_mode = QtWidgets.QComboBox()
        self.time_mode.addItem('SDO: nominal filename clocks', 'nominal')
        self.time_mode.addItem('SDO: convert filename clocks to UTC', 'utc')
        grid_bar.addWidget(self.time_mode)
        self.tolerance = QtWidgets.QDoubleSpinBox()
        self.tolerance.setRange(0, 86400)
        self.tolerance.setValue(60)
        self.tolerance.setSuffix(' s')
        grid_bar.addWidget(QtWidgets.QLabel('Max |Δt|'))
        grid_bar.addWidget(self.tolerance)
        grid_bar.addStretch()
        timeline = QtWidgets.QHBoxLayout()
        outer.addLayout(timeline)
        self.play = QtWidgets.QPushButton('Play')
        self.play.setCheckable(True)
        self.play.toggled.connect(self.toggle_play)
        previous = QtWidgets.QPushButton('◀')
        following = QtWidgets.QPushButton('▶')
        previous.clicked.connect(lambda: self.step(-1))
        following.clicked.connect(lambda: self.step(1))
        self.time = QtWidgets.QComboBox()
        self.time.addItems([f'{i+1}/{store.nt}  {t.isoformat(timespec="milliseconds")} UTC'
                           for i, t in enumerate(store.times)])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, store.nt - 1)
        self.fps = QtWidgets.QDoubleSpinBox()
        self.fps.setRange(.1, 30)
        self.fps.setValue(3)
        self.fps.setSuffix(' fps')
        self.loop = QtWidgets.QCheckBox('Loop')
        self.loop.setChecked(True)
        for control in (previous, self.play, following, self.time, self.slider, self.fps, self.loop):
            timeline.addWidget(control)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.fps.valueChanged.connect(lambda: self.timer.setInterval(round(1000 / self.fps.value())))
        self.time.currentIndexChanged.connect(self.set_time)
        self.slider.valueChanged.connect(self.set_time)
        self.time_mode.currentIndexChanged.connect(self.render)
        self.tolerance.valueChanged.connect(self.render)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.grid_host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_host)
        scroll.setWidget(self.grid_host)
        outer.addWidget(scroll, 1)
        if store.warnings:
            warning = QtWidgets.QLabel('\n'.join(store.warnings))
            warning.setWordWrap(True)
            outer.addWidget(warning)
        self.rebuild_grid()

    def rebuild_grid(self):
        count = self.rows.value() * self.columns.value()
        while len(self.panels) > count:
            panel = self.panels.pop()
            self.grid.removeWidget(panel)
            panel.deleteLater()
        while len(self.panels) < count:
            panel = ImagePanel(self, len(self.panels) + 1)
            self.panels.append(panel)
            panel.source.setCurrentIndex((len(self.panels) - 1) % len(self.store.sources))
        for panel in self.panels:
            self.grid.removeWidget(panel)
        for i, panel in enumerate(self.panels):
            self.grid.addWidget(panel, i // self.columns.value(), i % self.columns.value())

    def set_time(self, index):
        self.time_index = index
        for control in (self.time, self.slider):
            control.blockSignals(True)
        self.time.setCurrentIndex(index)
        self.slider.setValue(index)
        for control in (self.time, self.slider):
            control.blockSignals(False)
        self.render()

    def render(self, *_):
        for panel in self.panels:
            panel.render()

    def step(self, amount):
        self.set_time(max(0, min(self.store.nt - 1, self.time_index + amount)))

    def toggle_play(self, playing):
        self.play.setText('Pause' if playing else 'Play')
        if playing:
            if self.time_index == self.store.nt - 1:
                self.set_time(0)
            self.timer.start(round(1000 / self.fps.value()))
        else:
            self.timer.stop()

    def advance(self):
        if self.time_index + 1 < self.store.nt:
            self.step(1)
        elif self.loop.isChecked():
            self.set_time(0)
        else:
            self.play.setChecked(False)

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


class FileDialog(QtWidgets.QDialog):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle('Open GRIS comparison data')
        form = QtWidgets.QFormLayout(self)
        self.fields = {}
        for key, label in [('observation', 'Observation FITS (required)'), ('timestamps', 'Timestamp CSV (required)'),
                           ('atmosphere', 'Merged atmosphere (optional)'), ('profiles', 'Merged profiles (optional)'),
                           ('aligned_root', 'Aligned SDO root (optional)')]:
            line = QtWidgets.QLineEdit(str(getattr(args, key) or ''))
            self.fields[key] = line
            row = QtWidgets.QHBoxLayout()
            row.addWidget(line)
            browse = QtWidgets.QPushButton('Browse…')
            browse.clicked.connect(lambda checked=False, k=key: self.browse(k))
            row.addWidget(browse)
            form.addRow(label, row)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Open |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.resize(850, 240)

    def browse(self, key):
        if key == 'aligned_root':
            value = QtWidgets.QFileDialog.getExistingDirectory(self, 'Aligned SDO root')
        else:
            value, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select ' + key)
        if value:
            self.fields[key].setText(value)

    def paths(self):
        return {key: line.text().strip() or None for key, line in self.fields.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', help='actual_filepath_ca FITS cube')
    parser.add_argument('--timestamps', default=str(Path(__file__).with_name('serie_timestamps.csv')))
    parser.add_argument('--atmosphere', help='output_merged_atmos HDF5/.nc')
    parser.add_argument('--profiles', help='output_merged_profs HDF5/.nc')
    parser.add_argument('--aligned-root', help='Directory containing AIA/ and HMI/')
    parser.add_argument('--rows', type=int, choices=range(1, 7), default=2)
    parser.add_argument('--columns', type=int, choices=range(1, 7), default=2)
    parser.add_argument('--wave-start', type=float, default=8540.67304823)
    parser.add_argument('--wave-step', type=float, default=0.0109907)
    args = parser.parse_args()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    paths = {key: getattr(args, key) for key in ('observation', 'timestamps', 'atmosphere', 'profiles', 'aligned_root')}
    dialog = None
    while True:
        if not paths['observation'] or dialog is not None:
            dialog = dialog or FileDialog(args)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return 0
            paths = dialog.paths()
        try:
            store = DataStore(**paths, wave_start=args.wave_start, wave_step=args.wave_step)
            break
        except Exception as exc:
            QtWidgets.QMessageBox.critical(None, 'Cannot open data', str(exc))
            dialog = dialog or FileDialog(args)
    widget = ComparisonWidget(store, args.rows, args.columns)
    widget.setWindowTitle('GRIS · Inversion and SDO comparison')
    widget.resize(1400, 1000)
    widget.show()
    try:
        return app.exec()
    finally:
        store.close()


if __name__ == '__main__':
    sys.exit(main())
