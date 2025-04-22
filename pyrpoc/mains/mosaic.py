from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSpinBox,
    QComboBox, QPushButton, QScrollArea, QWidget, QCheckBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
from pyrpoc.mains import acquisition
from pyrpoc.helpers.prior_stage.functions import *
import numpy as np
import random

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
        self.layout = QVBoxLayout(self)

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)

        params_group = QGroupBox("Mosaic Parameters")
        params_layout = QGridLayout(params_group)

        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 100); self.rows_spin.setValue(3)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 100); self.cols_spin.setValue(3)
        self.overlap_spin = QSpinBox(); self.overlap_spin.setRange(0, 100); self.overlap_spin.setValue(10)
        self.pattern_combo = QComboBox(); self.pattern_combo.addItems(["Snake", "Raster"])
        self.fov_um_spin = QSpinBox(); self.fov_um_spin.setRange(1, 10000); self.fov_um_spin.setValue(100)
        self.grid_checkbox = QCheckBox("Show Tile Grid"); self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.update_display)

        params_layout.addWidget(QLabel("Rows:"), 0, 0)
        params_layout.addWidget(self.rows_spin, 0, 1)
        params_layout.addWidget(QLabel("Columns:"), 0, 2)
        params_layout.addWidget(self.cols_spin, 0, 3)
        params_layout.addWidget(QLabel("Overlap (%):"), 1, 0)
        params_layout.addWidget(self.overlap_spin, 1, 1)
        params_layout.addWidget(QLabel("Pattern:"), 2, 0)
        params_layout.addWidget(self.pattern_combo, 2, 1)
        params_layout.addWidget(QLabel("FOV Size (μm):"), 3, 0)
        params_layout.addWidget(self.fov_um_spin, 3, 1)
        params_layout.addWidget(self.grid_checkbox, 4, 0, 1, 2)

        self.layout.addWidget(params_group)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.display_label = ZoomableLabel(scroll_area=self.scroll_area)
        self.scroll_area.setWidget(self.display_label)
        self.layout.addWidget(self.scroll_area)

        self.start_button = QPushButton("Start Mosaic Imaging")
        self.start_button.setAutoDefault(False)
        self.start_button.setDefault(False)
        self.start_button.clicked.connect(self.prepare_run)
        self.layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_mosaic)
        self.layout.addWidget(self.cancel_button)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.resize(1200, 800)

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
        self.cancelled = False
        self._rows = self.rows_spin.value()
        self._cols = self.cols_spin.value()
        self._overlap = self.overlap_spin.value() / 100.0
        self._pattern = self.pattern_combo.currentText()
        self._fov_um = self.fov_um_spin.value()

        try:
            self._port = int(self.main_gui.prior_port_entry.get().strip())
            self._x0, self._y0 = get_xy(self._port)
        except Exception as e:
            self.update_status(f"Stage error: {e}")
            return

        acquisition.acquire(self.main_gui, auxilary=True)
        data = getattr(self.main_gui, 'data', None)
        if not data or not isinstance(data, (list, tuple)) or len(data) == 0:
            self.update_status("Acquisition failed.")
            return

        self._tile_h, self._tile_w = data[0].shape
        step_px = int(self._tile_w * (1 - self._overlap))
        step_um = int(self._fov_um * (1 - self._overlap))

        mosaic_h = step_px * (self._rows - 1) + self._tile_h
        mosaic_w = step_px * (self._cols - 1) + self._tile_w
        self.mosaic_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float32)

        self._tile_order = []
        self._tile_colors = {}
        for i in range(self._rows):
            col_range = range(self._cols) if (i % 2 == 0 or self._pattern == "Raster") else reversed(range(self._cols))
            for j in col_range:
                dx_um = int(j * step_um)
                dy_um = int(i * step_um)
                dx_px = int(j * step_px)
                dy_px = int(i * step_px)
                self._tile_order.append((i, j, dx_um, dy_um, dx_px, dy_px))
                self._tile_colors[(i, j)] = QColor(*[random.randint(180, 255) for _ in range(3)], 220)

        self.update_status("Starting mosaic...")
        QTimer.singleShot(0, self.process_next_tile)

    def process_next_tile(self):
        if self.cancelled or not self._tile_order:
            self.update_status("Mosaic acquisition complete.")
            self.update_display()
            return

        i, j, dx_um, dy_um, dx_px, dy_px = self._tile_order.pop(0)
        try:
            move_xy(self._port, self._x0 + dx_um, self._y0 + dy_um)
        except Exception as e:
            self.update_status(f"Move failed: {e}")
            return

        self.update_status(f"Acquiring tile ({i+1}, {j+1})...")
        try:
            acquisition.acquire(self.main_gui, auxilary=True)
        except Exception as e:
            self.update_status(f"Acquisition failed: {e}")
            return

        self.blend_tile(i, j, dx_px, dy_px)
        QTimer.singleShot(100, self.process_next_tile)

    def blend_tile(self, i, j, dx, dy):
        data = getattr(self.main_gui, 'data', None)
        if not data or not isinstance(data, (list, tuple)) or len(data) == 0:
            return

        visibility = getattr(self.main_gui, 'image_visibility', [True] * len(data))
        colors = getattr(self.main_gui, 'image_colors', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

        tile_rgb = np.zeros((self._tile_h, self._tile_w, 3), dtype=np.float32)

        for ch, frame in enumerate(data):
            if ch < len(visibility) and visibility[ch]:
                vmin, vmax = np.min(frame), np.max(frame)
                norm = (frame - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(frame)
                for c in range(3):
                    tile_rgb[..., c] += norm * (colors[ch][c] / 255.0)

        tile_rgb = np.clip(tile_rgb, 0, 1)

        y1, y2 = dy, dy + self._tile_h
        x1, x2 = dx, dx + self._tile_w

        if y2 > self.mosaic_rgb.shape[0] or x2 > self.mosaic_rgb.shape[1]:
            print(f"Tile ({i},{j}) exceeds mosaic bounds.")
            return

        self.mosaic_rgb[y1:y2, x1:x2] = (self.mosaic_rgb[y1:y2, x1:x2] + tile_rgb) / 2.0
        self.update_display()

    def update_display(self):
        rgb_uint8 = (np.clip(self.mosaic_rgb, 0, 1) * 255).astype(np.uint8)
        h, w, _ = rgb_uint8.shape
        image = QImage(rgb_uint8.data, w, h, QImage.Format_RGB888)

        if self.grid_checkbox.isChecked():
            painter = QPainter(image)
            for (i, j), color in self._tile_colors.items():
                step_px = int(self._tile_w * (1 - self._overlap))
                dx = int(j * step_px)
                dy = int(i * step_px)
                pen = QPen(color)
                pen.setWidth(3)
                pen.setStyle(Qt.DashLine if self._overlap > 0 else Qt.SolidLine)
                painter.setPen(pen)
                painter.drawRect(dx, dy, self._tile_w, self._tile_h)
            painter.end()

        pixmap = QPixmap.fromImage(image)
        self.display_label.setPixmap(pixmap)
        self.display_label.repaint_scaled()
