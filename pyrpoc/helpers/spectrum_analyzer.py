import sys
import numpy as np
import csv
import json
import os
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QGroupBox, QGridLayout, QCheckBox, QInputDialog, QComboBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg
from pyrpoc.helpers.utils import generate_data
from pyrpoc.mains import acquisition

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'metadata', 'spectral_calibration.json')

class SpectrumAnalyzer(QDialog):
    def __init__(self, gui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrum Analyzer")
        self.gui = gui
        self.setMinimumSize(900, 600)
        self.running = False
        self.calibration = self.load_default_calibration()
        self.calibration_points = []
        self.calibrating = False
        self.use_wavenumber = True

        self.layout = QVBoxLayout(self)

        self.settings_group = QGroupBox("Scan Parameters")
        self.settings_layout = QGridLayout(self.settings_group)

        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 200); self.steps_spin.setValue(10)
        self.settings_layout.addWidget(QLabel("Number of Steps:"), 0, 0)
        self.settings_layout.addWidget(self.steps_spin, 0, 1)

        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 100000); self.start_spin.setValue(0)
        self.settings_layout.addWidget(QLabel("Start Delay (µm):"), 1, 0)
        self.settings_layout.addWidget(self.start_spin, 1, 1)

        self.stop_spin = QSpinBox(); self.stop_spin.setRange(0, 100000); self.stop_spin.setValue(100)
        self.settings_layout.addWidget(QLabel("Stop Delay (µm):"), 2, 0)
        self.settings_layout.addWidget(self.stop_spin, 2, 1)

        self.sim_check = QCheckBox("Simulate")
        self.sim_check.setChecked(self.gui.simulation_mode.get())
        self.settings_layout.addWidget(self.sim_check, 3, 0, 1, 2)

        self.axis_mode_combo = QComboBox()
        self.axis_mode_combo.addItems(["Wavenumber", "Delay (µm)"])
        self.axis_mode_combo.currentTextChanged.connect(self.update_display_axis)
        self.settings_layout.addWidget(QLabel("X Axis Mode:"), 4, 0)
        self.settings_layout.addWidget(self.axis_mode_combo, 4, 1)

        self.layout.addWidget(self.settings_group)

        self.plot_widget = pg.PlotWidget(background='k')
        self.plot = self.plot_widget.plot(pen=pg.mkPen(color='c', width=2), symbol='o', symbolBrush='w', symbolPen=None)
        self.plot_widget.setLabel('bottom', 'Wavenumber (cm⁻¹)')
        self.plot_widget.setLabel('left', 'Average Intensity')
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('w'))
        self.plot_widget.getAxis('left').setPen(pg.mkPen('w'))
        self.plot_widget.getAxis('bottom').setTextPen('w')
        self.plot_widget.getAxis('left').setTextPen('w')
        self.plot_widget.scene().sigMouseClicked.connect(self.handle_mouse_click)
        self.layout.addWidget(self.plot_widget)

        self.button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Acquisition")
        self.stop_button = QPushButton("Stop")
        self.calibrate_button = QPushButton("Start Calibration")
        self.save_csv_button = QPushButton("Save CSV")
        self.save_cal_button = QPushButton("Save as Defaults")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.calibrate_button.clicked.connect(self.toggle_calibration)
        self.save_csv_button.clicked.connect(self.save_csv)
        self.save_cal_button.clicked.connect(self.save_default_calibration)

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.calibrate_button)
        self.button_layout.addWidget(self.save_csv_button)
        self.button_layout.addWidget(self.save_cal_button)
        self.layout.addLayout(self.button_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_step)

    def load_default_calibration(self):
        try:
            with open(CALIBRATION_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print("Warning: Failed to load default calibration:", e)
            return [(0, 4000), (100, 3000)]

    def delay_to_wavenumber(self, delay_um):
        if len(self.calibration) < 2:
            return delay_um
        (x0, y0), (x1, y1) = self.calibration[:2]
        m = (y1 - y0) / (x1 - x0)
        return y0 + m * (delay_um - x0)

    def start_acquisition(self):
        self.positions = np.linspace(
            self.start_spin.value(),
            self.stop_spin.value(),
            self.steps_spin.value()
        )
        self.wavenumbers = np.array([self.delay_to_wavenumber(p) for p in self.positions])
        self.current_step = 0
        self.spectrum_values = []
        self.running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.timer.start(200)

    def stop_acquisition(self):
        self.running = False
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_symbol_colors(self):
        symbol_brushes = []
        for k in range(len(self.spectrum_values)):
            if self.calibrating:
                match = any(np.isclose(self.positions[k], p[0], atol=1e-3) for p in self.calibration_points)
                color = (0, 255, 0) if match else (255, 0, 0)
            else:
                color = (180, 180, 180)
            symbol_brushes.append(pg.mkBrush(color))

        x_data = self.wavenumbers if self.use_wavenumber else self.positions

        self.plot.setData(
            x_data[:len(self.spectrum_values)],
            self.spectrum_values,
            symbol='o',
            symbolBrush=symbol_brushes,
            symbolSize=8
        )

    def next_step(self):
        if not self.running or self.current_step >= len(self.positions):
            self.stop_acquisition()
            return

        pos = self.positions[self.current_step]

        try:
            if not self.sim_check.isChecked():
                self.gui.zaber_stage.move_absolute_um(pos)
                self.gui.root.update_idletasks()
                acquisition.acquire(self.gui, auxilary=True)
                data = self.gui.data[0]
            else:
                wn = self.wavenumbers[self.current_step]
                intensity = 1000 * np.exp(-((wn - 2900) ** 2) / (2 * 40 ** 2)) + np.random.normal(0, 30)
                data = np.full((128, 128), intensity)
        except Exception as e:
            print("Acquisition error:", e)
            self.stop_acquisition()
            return

        avg_intensity = data.mean()
        self.spectrum_values.append(avg_intensity)

        self.update_symbol_colors()
        self.current_step += 1

    def handle_mouse_click(self, event):
        if not self.calibrating:
            return
        pos = event.scenePos()
        vb = self.plot_widget.plotItem.vb
        if vb.mapSceneToView(pos):
            mouse_x = vb.mapSceneToView(pos).x()
            x_data = self.wavenumbers if self.use_wavenumber else self.positions
            index = int(np.argmin(np.abs(np.array(x_data[:len(self.spectrum_values)]) - mouse_x)))
            if 0 <= index < len(self.spectrum_values):
                guess = x_data[index]
                new_wn, ok = QInputDialog.getDouble(self, "Assign Wavenumber", f"Assign true wavenumber for peak near {int(guess)}:", guess, 0, 10000, 2)
                if ok:
                    self.calibration_points.append((self.positions[index], new_wn))
                    print(f"Added calibration point: delay {self.positions[index]:.2f} µm → {new_wn:.2f} cm⁻¹")
                    self.update_symbol_colors()

    def toggle_calibration(self):
        if not self.calibrating:
            self.calibrating = True
            self.calibration_points.clear()
            self.calibrate_button.setText("Finish Calibration")
            print("Calibration mode ON. Click on spectrum peaks to assign known wavenumbers.")
        else:
            if len(self.calibration_points) >= 2:
                self.calibration = sorted(self.calibration_points, key=lambda p: p[0])[:2]
                print("Calibration updated:", self.calibration)
                self.wavenumbers = np.array([self.delay_to_wavenumber(p) for p in self.positions])
            else:
                print("Not enough calibration points to apply.")
            self.calibrating = False
            self.calibrate_button.setText("Start Calibration")
        self.update_symbol_colors()

    def update_display_axis(self):
        self.use_wavenumber = (self.axis_mode_combo.currentText() == "Wavenumber")
        self.plot_widget.setLabel('bottom', 'Wavenumber (cm⁻¹)' if self.use_wavenumber else 'Delay (µm)')
        self.update_symbol_colors()

    def save_csv(self):
        if not self.spectrum_values:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Spectrum CSV", "spectrum.csv", "CSV Files (*.csv)")
        if path:
            x_vals = self.wavenumbers if self.use_wavenumber else self.positions
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["X", "Intensity"])
                for x, y in zip(x_vals, self.spectrum_values):
                    writer.writerow([x, y])
            print(f"Spectrum saved to {path}")

    def save_default_calibration(self):
        try:
            os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
            with open(CALIBRATION_PATH, 'w') as f:
                json.dump(self.calibration, f, indent=2)
            print(f"Default calibration saved to {CALIBRATION_PATH}")
        except Exception as e:
            print("Error saving default calibration:", e)
