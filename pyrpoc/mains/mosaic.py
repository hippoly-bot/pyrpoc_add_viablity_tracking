from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QScrollArea, QWidget, QCheckBox, QFileDialog,
    QHBoxLayout, QLineEdit, QDoubleSpinBox, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRunnable, QThread, QThreadPool, QObject, pyqtSignal, pyqtSlot
from pyrpoc.mains import acquisition
from pyrpoc.helpers.prior_stage.functions import *
import numpy as np
import random
import json
import os
from PIL import Image

class MosaicDialog(QDialog):
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mosaic Imaging")
        self.main_gui = main_gui
        self.cancelled = False
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(1200, 800)

        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)

        self.sidebar_widget = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar_widget)
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(12)
        self.sidebar_widget.setFixedWidth(300)

        self.save_group = QGroupBox("Save Options")
        save_layout = QGridLayout(self.save_group)

        save_layout.addWidget(QLabel("Save Folder:"), 0, 0)
        self.save_folder_entry = QLineEdit()
        self.save_folder_entry.setPlaceholderText("Select folder...")
        save_layout.addWidget(self.save_folder_entry, 0, 1)
        browse_btn = QPushButton("📂")
        browse_btn.clicked.connect(self.browse_save_folder)
        save_layout.addWidget(browse_btn, 0, 2)

        self.save_metadata_checkbox = QCheckBox("Save Metadata (.json)")
        self.save_stitched_checkbox = QCheckBox("Save Stitched TIFF")
        self.save_tiles_checkbox = QCheckBox("Save Individual Tiles")

        self.save_metadata_checkbox.setChecked(True)
        self.save_stitched_checkbox.setChecked(True)
        self.save_tiles_checkbox.setChecked(True)

        save_layout.addWidget(self.save_metadata_checkbox, 1, 0, 1, 3)
        save_layout.addWidget(self.save_stitched_checkbox, 2, 0, 1, 3)
        save_layout.addWidget(self.save_tiles_checkbox, 3, 0, 1, 3)

        params_group = QGroupBox("Mosaic Parameters")
        params_layout = QGridLayout(params_group)

        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 1000); self.rows_spin.setValue(3)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 1000); self.cols_spin.setValue(3)
        self.overlap_spin = QSpinBox(); self.overlap_spin.setRange(0, 100); self.overlap_spin.setValue(10)
        self.repetitions_spin = QSpinBox(); self.repetitions_spin.setRange(1, 100); self.repetitions_spin.setValue(1)
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(["Snake", "Raster"])
        self.fov_um_spin = QSpinBox(); self.fov_um_spin.setRange(1, 10000); self.fov_um_spin.setValue(100)
        self.repetitions_spin = QSpinBox(); self.repetitions_spin.setRange(1,100); self.repetitions_spin.setValue(1)
        self.display_mosaic_checkbox = QCheckBox("Display Mosaic Live")
        self.display_mosaic_checkbox.setChecked(True)
        

        self.rows_spin.valueChanged.connect(self.report_memory_estimate)
        self.cols_spin.valueChanged.connect(self.report_memory_estimate)

        
        params_layout.addWidget(QLabel("Rows:"), 0, 0)
        params_layout.addWidget(self.rows_spin, 0, 1)
        params_layout.addWidget(QLabel("Columns:"), 1, 0)
        params_layout.addWidget(self.cols_spin, 1, 1)
        params_layout.addWidget(QLabel("Overlap (%):"), 2, 0)
        params_layout.addWidget(self.overlap_spin, 2, 1)
        params_layout.addWidget(QLabel('Repetitions per Tile'), 3, 0)
        params_layout.addWidget(self.overlap_spin, 3, 1)
        params_layout.addWidget(QLabel("Pattern:"), 4, 0)
        params_layout.addWidget(self.pattern_combo, 4, 1)
        params_layout.addWidget(QLabel("FOV Size (μm):"), 5, 0)
        params_layout.addWidget(self.fov_um_spin, 5, 1)
        params_layout.addWidget(self.display_mosaic_checkbox, 6, 0, 1, 2)

        self.start_button = QPushButton("Start Mosaic Imaging")
        self.start_button.setAutoDefault(False)
        self.start_button.setDefault(False)
        self.start_button.clicked.connect(self.prepare_run)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_mosaic)

        self.af_group = QGroupBox("Autofocus Settings")
        af_layout = QGridLayout(self.af_group)

        self.af_enabled_checkbox = QCheckBox("Enable autofocus")
        self.af_enabled_checkbox.setChecked(True)

        self.af_every_n_label = QLabel("Tiles per autofocus:")
        self.af_every_n_spin = QSpinBox()
        self.af_every_n_spin.setRange(1, 100)
        self.af_every_n_spin.setValue(1)

        self.af_max_steps_label = QLabel("Max steps per autofocus:")
        self.af_max_steps_spin = QSpinBox()
        self.af_max_steps_spin.setRange(1, 100)
        self.af_max_steps_spin.setValue(5)

        self.af_stepsize_label = QLabel("Step Size (μm):")
        self.af_stepsize_spin = QDoubleSpinBox()
        self.af_stepsize_spin.setDecimals(1)
        self.af_stepsize_spin.setRange(0.1, 10.0)
        self.af_stepsize_spin.setValue(0.1)
        self.af_stepsize_spin.setSingleStep(0.1)

        af_layout.addWidget(self.af_enabled_checkbox, 0, 0, 1, 2)
        af_layout.addWidget(self.af_every_n_label, 1, 0); af_layout.addWidget(self.af_every_n_spin, 1, 1)
        af_layout.addWidget(self.af_max_steps_label, 2, 0); af_layout.addWidget(self.af_max_steps_spin, 2, 1)
        af_layout.addWidget(self.af_stepsize_label, 3, 0); af_layout.addWidget(self.af_stepsize_spin, 3, 1)

        self.sidebar_layout.addWidget(self.save_group)
        self.sidebar_layout.addWidget(params_group)
        self.sidebar_layout.addWidget(self.af_group)
        self.sidebar_layout.addWidget(self.start_button)
        self.sidebar_layout.addWidget(self.cancel_button)
        self.sidebar_layout.addStretch()
        main_layout.addWidget(self.sidebar_widget)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.view, stretch=1)
        right_layout.addWidget(self.status_label)

        main_layout.addLayout(right_layout, stretch=1)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def update_status(self, text):
        QTimer.singleShot(0, lambda: self.status_label.setText(text))

    def cancel_mosaic(self):
        self.cancelled = True
        self.update_status("Mosaic acquisition cancelled.")

    def prepare_run(self):
        self.scene.clear()

        self.tile_w = self.main_gui.config['numsteps_x']
        self.tile_h = self.main_gui.config['numsteps_y']
        overlap = self.overlap_spin.value() / 100.0
        self.step_px = int(self.tile_w * (1 - overlap))

        rows, cols = self.rows_spin.value(), self.cols_spin.value()
        self._tile_order = []
        for i in range(rows):
            cols_range = range(cols) if (i % 2 == 0) else reversed(range(cols))
            for j in cols_range:
                dx_px = j * self.step_px
                dy_px = i * self.step_px
                self._tile_order.append((i, j, dx_px, dy_px))

        self.worker = MosaicWorker(
            gui=self.main_gui,
            tile_order=self._tile_order,
            port=int(self.main_gui.prior_port_entry.get().strip()),
            af_enabled=self.af_enabled_checkbox.isChecked(),
            af_interval=self.af_every_n_spin.value(),
            af_stepsize=self.af_stepsize_spin.value(),
            af_max_steps=self.af_max_steps_spin.value(),
            simulate=self.main_gui.simulation_mode.get(),
            chan=self.main_gui.config["channel_names"][0],
            repetitions=self.repetitions_spin.value()
        )
        self.worker.tile_ready.connect(self.on_tile_ready)
        self.worker.finished.connect(self.on_mosaic_complete)
        self.worker.error.connect(lambda msg: self.update_status(f"Error: {msg}"))
        self.worker.start()
        self.update_status("Starting mosaic...")

    @pyqtSlot(int, int, int, int, object)
    def on_tile_ready(self, i, j, dx_px, dy_px, data):
        self.update_status(f"Acquired tile ({i+1},{j+1})...")
        if self.save_tiles_checkbox.isChecked():
            for ch, frame in enumerate(data):
                Image.fromarray(frame).save(os.path.join(self.save_folder_entry.text(), f"tile_{i}_{j}_ch{ch}.tif"))

        tile_rgb = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.float32)
        visibility = getattr(self.main_gui, 'image_visibility', [True]*len(data))
        colors = getattr(self.main_gui, 'image_colors', [(255,0,0),(0,255,0),(0,0,255)])
        for ch, frame in enumerate(data):
            if ch < len(visibility) and visibility[ch]:
                norm = frame.astype(np.float32)
                for c in range(3):
                    tile_rgb[..., c] += norm * (colors[ch][c]/255.0)
        tile_rgb = np.clip(tile_rgb, 0, 1)

        rgb8 = (tile_rgb * 255).astype(np.uint8)
        h, w = rgb8.shape[:2]
        image = QImage(rgb8.data, w, h, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        item = QGraphicsPixmapItem(pixmap)
        item.setPos(dx_px, dy_px)
        self.scene.addItem(item)

    @pyqtSlot()
    def on_mosaic_complete(self):
        self.update_status("Mosaic complete.")
        if self.save_stitched_checkbox.isChecked():
            rect = self.scene.sceneRect()
            out = QImage(int(rect.width()), int(rect.height()), QImage.Format_RGB888)
            painter = QPainter(out)
            self.scene.render(painter)
            painter.end()
            out.save(os.path.join(self.save_folder_entry.text(), "stitched_mosaic.tif"))
        if self.save_metadata_checkbox.isChecked():
            md = {
                'rows': self.rows_spin.value(),
                'cols': self.cols_spin.value(),
                'overlap': self.overlap_spin.value(),
                'order': [(i, j) for i,j,_,_ in self._tile_order]
            }
            with open(os.path.join(self.save_folder_entry.text(), 'mosaic_metadata.json'), 'w') as f:
                json.dump(md, f, indent=2)
        self.update_status("Saved mosaic and metadata.")

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder: self.save_folder_entry.setText(folder)

class MosaicWorker(QThread):
    tile_ready = pyqtSignal(int,int,int,int,object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, gui, port, x0, y0, tile_order, tile_repetitions,
                 af_enabled, af_interval, af_stepsize, af_max_steps,
                 simulate, chan):
        super().__init__()
        self.gui = gui; self.port = port; self.x0 = x0; self.y0 = y0
        self.tile_order = tile_order; self.tile_repetitions = tile_repetitions
        self.af_enabled = af_enabled; self.af_interval = af_interval
        self.af_stepsize = af_stepsize; self.af_max_steps = af_max_steps
        self.simulate = simulate; self.chan = chan

    def run(self):
        try:
            for idx,(i,j,dx_um,dy_um,dx_px,dy_px) in enumerate(self.tile_order):
                for _ in range(self.tile_repetitions):
                    acquisition.acquire(self.gui, auxilary=True)
                data = getattr(self.gui,'data',[]) or []
                self.tile_ready.emit(i,j,dx_px,dy_px,data)
                
                move_xy(self.port, self.x0+dx_um, self.y0-dy_um)
                if self.af_enabled and idx % self.af_interval == 0 and not self.simulate:
                    auto_focus(self.gui, self.port, self.chan,
                               step_size=self.af_stepsize, numsteps=self.af_max_steps)
                
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
