from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QCheckBox, QLabel, QMenu, QGraphicsTextItem, 
    QAction, QDialog, QGroupBox, QGridLayout, QSpinBox,
    QComboBox, QScrollArea
)
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QBrush, QPainter, QFont, QColor, QPalette, QImage
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QVariant
from superqt import QRangeSlider
from pyrpoc.mains import acquisition
import numpy as np
import threading
import time

class MosaicDialog(QDialog):
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mosaic Imaging")
        self.main_gui = main_gui
        self.layout = QVBoxLayout(self)

        params_group = QGroupBox("Mosaic Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Rows:"), 0, 0)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 100)
        self.rows_spin.setValue(3)
        params_layout.addWidget(self.rows_spin, 0, 1)

        params_layout.addWidget(QLabel("Columns:"), 0, 2)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 100)
        self.cols_spin.setValue(3)
        params_layout.addWidget(self.cols_spin, 0, 3)

        params_layout.addWidget(QLabel("Overlap (%):"), 1, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 100)
        self.overlap_spin.setValue(10)
        params_layout.addWidget(self.overlap_spin, 1, 1)

        params_layout.addWidget(QLabel("Pattern:"), 2, 0)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Snake", "Raster"])
        params_layout.addWidget(self.pattern_combo, 2, 1)

        self.layout.addWidget(params_group)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll_area.setWidget(self.grid_widget)
        self.layout.addWidget(self.scroll_area)

        self.tile_labels = {}  

        self.start_button = QPushButton("Start Mosaic Imaging")
        self.start_button.clicked.connect(lambda: threading.Thread(target=self.run_mosaic, daemon=True).start())
        self.layout.addWidget(self.start_button)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

        self.initialize_grid()

        self.rows_spin.valueChanged.connect(self.initialize_grid)
        self.cols_spin.valueChanged.connect(self.initialize_grid)

    def initialize_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.tile_labels.clear()

        rows = self.rows_spin.value()
        cols = self.cols_spin.value()

        for i in range(rows):
            for j in range(cols):
                placeholder = QLabel()
                placeholder.setFixedSize(256, 256)
                placeholder.setStyleSheet("background-color: #222; border: 1px solid #444;")
                placeholder.setAlignment(Qt.AlignCenter)
                placeholder.setText(f"{i},{j}")
                self.grid_layout.addWidget(placeholder, i, j)
                self.tile_labels[(i, j)] = placeholder

    def run_mosaic(self):
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        overlap = self.overlap_spin.value() / 100.0

        try:
            x0, y0 = map(float, self.main_gui.prior_pos_entry.get().split(","))
        except Exception:
            self.status_label.setText("Invalid initial XY position")
            return

        fov_x = fov_y = 100
        step_x = fov_x * (1 - overlap)
        step_y = fov_y * (1 - overlap)

        for i in range(rows):
            cols_order = range(cols) if (i % 2 == 0 or self.pattern_combo.currentText() == "Raster") else reversed(range(cols))
            for j in cols_order:
                dx = j * step_x
                dy = i * step_y
                new_x = x0 + dx
                new_y = y0 + dy

                self.status_label.setText(f"Moving to tile ({i+1}, {j+1})...")
                self.main_gui.prior_pos_entry.delete(0, 'end')
                self.main_gui.prior_pos_entry.insert(0, f"{new_x:.1f}, {new_y:.1f}")
                self.main_gui.move_prior_stage_xy()
                time.sleep(0.3)

                self.status_label.setText(f"Acquiring tile ({i+1}, {j+1})...")
                acquisition.acquire(self.main_gui, auxilary=True)

                self.display_tile(i, j)

        self.status_label.setText("Mosaic acquisition complete.")

    def display_tile(self, i, j):
        layers = getattr(self.main_gui, 'image_layers', [])
        visibility = getattr(self.main_gui, 'image_visibility', [True] * len(layers))
        colors = getattr(self.main_gui, 'image_colors', [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

        if not layers:
            return

        h, w = layers[0].shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for ch, layer in enumerate(layers):
            if ch < len(visibility) and visibility[ch]:
                norm = np.clip(layer.astype(np.float32) / 255.0, 0, 1)
                for c in range(3):
                    rgb[..., c] += norm * (colors[ch][c] / 255.0)

        rgb = np.clip(rgb, 0, 1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        h, w, _ = rgb_uint8.shape

        qimg = QImage(rgb_uint8.data, w, h, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        label = self.tile_labels.get((i, j))
        if label:
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
