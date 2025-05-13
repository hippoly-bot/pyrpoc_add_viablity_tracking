from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QScrollArea, QWidget, QCheckBox, QFileDialog,
    QHBoxLayout, QLineEdit, QDoubleSpinBox
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

class ZoomableLabel(QLabel):
    def __init__(self, scroll_area=None):
        super().__init__()
        self.scroll_area = scroll_area
        self._pixmap = None
        self.scale_factor = 1.0
        self._drag_pos = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.repaint_scaled()

    def wheelEvent(self, event):
        if not self._pixmap:
            return

        old_pos = event.pos()
        old_scroll_x = self.scroll_area.horizontalScrollBar().value()
        old_scroll_y = self.scroll_area.verticalScrollBar().value()

        offset_x = old_pos.x() + old_scroll_x
        offset_y = old_pos.y() + old_scroll_y

        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        new_scale = self.scale_factor * factor
        new_scale = max(0.1, min(new_scale, 20))

        if new_scale == self.scale_factor:
            return

        self.scale_factor = new_scale
        self.repaint_scaled()

        new_scroll_x = int(offset_x * factor - old_pos.x())
        new_scroll_y = int(offset_y * factor - old_pos.y())

        self.scroll_area.horizontalScrollBar().setValue(new_scroll_x)
        self.scroll_area.verticalScrollBar().setValue(new_scroll_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and self.scroll_area:
            diff = event.pos() - self._drag_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - diff.x())
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - diff.y())
            self._drag_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def repaint_scaled(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self._pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


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
        self.grid_checkbox = QCheckBox("Show Tile Grid"); self.grid_checkbox.setChecked(True)
        self.display_mosaic_checkbox = QCheckBox("Display Mosaic Live")
        self.display_mosaic_checkbox.setChecked(True)
        
        
        self.grid_checkbox.stateChanged.connect(self.update_display)
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
        params_layout.addWidget(self.grid_checkbox, 6, 0, 1, 2)
        params_layout.addWidget(self.display_mosaic_checkbox, 7, 0, 1, 2)

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

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_label = ZoomableLabel(scroll_area=self.scroll_area)
        self.scroll_area.setWidget(self.display_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.scroll_area)
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
        self.save_dir = self.save_folder_entry.text().strip()
        if not self.save_dir:
            self.update_status("Please select a folder to save results.")
            return
        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_tiles_checkbox.isChecked():
            self.tile_dir = os.path.join(self.save_dir, "tiles")
            os.makedirs(self.tile_dir, exist_ok=True)

        self.cancelled = False
        self._rows = self.rows_spin.value()
        self._cols = self.cols_spin.value()
        self._overlap = self.overlap_spin.value() / 100.0
        self._repetitions = self.repetitions_spin.value()
        self._pattern = self.pattern_combo.currentText()
        self._fov_um = self.fov_um_spin.value()
        self._af_enabled = self.af_enabled_checkbox.isChecked()
        self._af_interval = self.af_every_n_spin.value()
        self._af_max_steps = self.af_max_steps_spin.value()
        self._af_stepsize = int(10 * self.af_stepsize_spin.value())
        self._simulate = self.main_gui.simulation_mode.get()
        self._chan = self.main_gui.config["channel_names"][0]

        # Initial stage position
        try:
            port = int(self.main_gui.prior_port_entry.text().strip())
            x0, y0 = get_xy(port)
            self._port, self._x0, self._y0 = port, x0, y0
        except Exception as e:
            self.update_status(f"Stage error: {e}")
            return

        # Use config to get tile dimensions instead of an initial acquisition
        self.tile_w = self.main_gui.config.get('numsteps_x')
        self.tile_h = self.main_gui.config.get('numsteps_y')
        self.step_px = int(self.tile_w * (1 - self._overlap))
        self.step_um = int(self._fov_um * (1 - self._overlap))

        # Build tile order & colors
        self._tile_order = []
        self._tile_colors = {}
        for i in range(self._rows):
            cols = range(self._cols) if (i % 2 == 0 or self._pattern == "Raster") else reversed(range(self._cols))
            for j in cols:
                dx_px = j * self.step_px
                dy_px = i * self.step_px
                dx_um = j * self.step_um
                dy_um = i * self.step_um
                self._tile_order.append((i, j, dx_um, dy_um, dx_px, dy_px))
                self._tile_colors[(i, j)] = QColor(*[random.randint(180,255) for _ in range(3)], 220)

        # Prepare mosaic arrays
        mosaic_h = self.step_px * (self._rows - 1) + self.tile_h
        mosaic_w = self.step_px * (self._cols - 1) + self.tile_w
        self.mosaic_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float32)
        self.mosaic_weight = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

        # Launch worker
        self.worker = MosaicWorker(
            gui=self.main_gui,
            port=self._port,
            x0=self._x0,
            y0=self._y0,
            tile_order=self._tile_order,
            tile_repetitions=self._repetitions,
            af_enabled=self._af_enabled,
            af_interval=self._af_interval,
            af_stepsize=self._af_stepsize,
            af_max_steps=self._af_max_steps,
            simulate=self._simulate,
            chan=self._chan
        )
        self.worker.tile_ready.connect(self.on_tile_ready)
        self.worker.finished.connect(self.on_mosaic_complete)
        self.worker.error.connect(lambda msg: self.update_status(f"Mosaic error: {msg}"))
        self.worker.start()
        self.update_status("Starting mosaic...")

    @pyqtSlot(int, int, int, int, object)
    def on_tile_ready(self, i, j, dx_px, dy_px, data):
        self.update_status(f"Acquired tile ({i+1},{j+1})...")
        # Save tiles
        if self.save_tiles_checkbox.isChecked():
            for ch, frame in enumerate(data):
                Image.fromarray(frame).save(os.path.join(self.tile_dir, f"tile_{i}_{j}_ch{ch}.tif"))
        # Blend
        tile_rgb = np.zeros((self.tile_h, self.tile_w, 3), dtype=np.float32)
        visibility = getattr(self.main_gui, 'image_visibility', [True]*len(data))
        colors = getattr(self.main_gui, 'image_colors', [(255,0,0),(0,255,0),(0,0,255)])
        for ch, frame in enumerate(data):
            if ch < len(visibility) and visibility[ch]:
                norm = frame.astype(np.float32)
                for c in range(3): tile_rgb[..., c] += norm * (colors[ch][c]/255.0)
        tile_rgb = np.clip(tile_rgb, 0, 1)
        y1, y2 = dy_px, dy_px + self.tile_h
        x1, x2 = dx_px, dx_px + self.tile_w
        self.mosaic_rgb[y1:y2, x1:x2] += tile_rgb
        self.mosaic_weight[y1:y2, x1:x2] += 1.0
        self.update_display()

    @pyqtSlot()
    def on_mosaic_complete(self):
        self.update_status("Mosaic acquisition complete.")
        self.update_display()
        self.save_mosaic()

    def update_display(self):
        weight = np.maximum(self.mosaic_weight[..., np.newaxis], 1.0)
        norm_rgb = self.mosaic_rgb / weight
        rgb8 = np.clip(norm_rgb*255,0,255).astype(np.uint8)
        h, w, _ = rgb8.shape
        image = QImage(rgb8.data, w, h, QImage.Format_RGB888)
        painter = QPainter(image)
        if self.grid_checkbox.isChecked():
            pen = QPen(); pen.setWidth(2); pen.setStyle(Qt.DashLine)
            for (i,j), color in self._tile_colors.items():
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawRect(j*self.step_px, i*self.step_px, self.tile_w, self.tile_h)
        painter.end()
        pix = QPixmap.fromImage(image)
        self.display_label.setPixmap(pix)
        self.display_label.repaint_scaled()

    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder: self.save_folder_entry.setText(folder)

    def save_mosaic(self):
        if self.save_metadata_checkbox.isChecked():
            md = {"rows":self._rows, "cols":self._cols, "overlap":self._overlap,
                  "pattern":self._pattern, "fov_um":self._fov_um,
                  "initial_position":(self._x0,self._y0),
                  "tile_order":[(i,j) for i,j,*_ in self._tile_order]}
            with open(os.path.join(self.save_dir,"mosaic_metadata.json"),"w") as f:
                json.dump(md, f, indent=2)
        if self.save_stitched_checkbox.isChecked():
            weight = np.maximum(self.mosaic_weight[...,np.newaxis],1.0)
            norm_rgb = self.mosaic_rgb / weight
            img = np.clip(norm_rgb*255,0,255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(self.save_dir,"stitched_mosaic.tif"))
        self.update_status("Mosaic saved.")

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
