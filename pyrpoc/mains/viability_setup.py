from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QVBoxLayout,
    QSpinBox,
    QWidget,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QMenu,
    QGraphicsTextItem,
    QAction,
    QDialog,
)
from PyQt5.QtGui import (
    QPixmap,
    QPainterPath,
    QPen,
    QBrush,
    QPainter,
    QFont,
    QColor,
    QPalette,
    QImage,
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import tifffile
import cv2
import os
import sys


class ThresholdSetupWindow(QWidget):
    def __init__(self, mode="live", callback=None):
        super().__init__()
        self.callback = callback
        self.setWindowTitle(f"{mode.title()} Threshold Setup")
        self.mode = mode
        self.image_stack = None
        self.current_frame_index = None
        self.roi_path = None
        self.drawing = False
        self.path = None
        self.points = []
        self.avg_std = None
        self.std_all = []
        self.diff_sequence = []

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Load
        load_btn = QPushButton("Load File")
        load_btn.clicked.connect(self.load_image_stack)
        main_layout.addWidget(load_btn)

        self.status_label = QLabel("No file loaded.")
        main_layout.addWidget(self.status_label)

        # ROI
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setMouseTracking(True)
        self.view.viewport().installEventFilter(self)
        main_layout.addWidget(self.view)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_frame)
        main_layout.addWidget(self.slider)

        ROI_layout = QHBoxLayout()
        self.check_btn = QPushButton("Check ROI")
        self.check_btn.clicked.connect(self.check_ROI)
        ROI_layout.addWidget(self.check_btn)

        self.use_roi_checkbox = QCheckBox("Use ROI")
        self.use_roi_checkbox.setChecked(True)
        ROI_layout.addWidget(self.use_roi_checkbox)
        main_layout.addLayout(ROI_layout)

        # Before subtraction settings
        main_layout.addWidget(QLabel("Start Frame:"))
        self.start_spin = QSpinBox()
        self.start_spin.setMinimum(1)
        main_layout.addWidget(self.start_spin)

        main_layout.addWidget(QLabel("Subtraction Frames Window (Δn):"))
        self.dt_spin = QSpinBox()
        self.dt_spin.setMinimum(1)
        self.dt_spin.setValue(30)
        main_layout.addWidget(self.dt_spin)

        # Compute, check and ensure buttons
        self.compute_btn = QPushButton("Compute STD")
        self.compute_btn.clicked.connect(self.compute_std)
        main_layout.addWidget(self.compute_btn)

        self.check_btn = QPushButton("Check Result")
        self.check_btn.clicked.connect(self.check_std)
        main_layout.addWidget(self.check_btn)

        self.use_result_btn = QPushButton("Use Result")
        self.use_result_btn.clicked.connect(self.use_result)
        main_layout.addWidget(self.use_result_btn)

        self.setLayout(main_layout)

    def load_image_stack(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select File(s)", "", "TIFF (*.tif);;Text (*.txt);;All Files (*)"
        )
        if not file_paths:
            return

        # Check extension of the first file to determine type
        first_file = file_paths[0].lower()
        if first_file.endswith(".txt"):
            if len(file_paths) != 1:
                self.status_label.setText("Please select a single text file.")
                return
            self.image_stack = np.loadtxt(first_file, delimiter=",")
            if self.image_stack.ndim == 2:
                self.image_stack = self.image_stack[np.newaxis, ...]
        elif first_file.endswith(".tif"):  # Assume multiple TIFFs
            stack = []
            for path in file_paths:
                img = tifffile.imread(path)
                if img.ndim == 2:
                    stack.append(img)
                elif img.ndim == 3:
                    stack.extend([frame for frame in img])
            self.image_stack = np.stack(stack, axis=0)

        self.status_label.setText(f"Loaded shape: {self.image_stack.shape}")
        self.current_frame_index = 0
        self.slider.setRange(0, self.image_stack.shape[0] - 1)
        self.slider.setValue(0)
        self.update_frame(0)

    def update_frame(self, frame_index):
        if self.image_stack is None:
            return

        self.current_frame_index = frame_index
        frame = self.image_stack[frame_index]
        height, width = frame.shape
        qimg = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, width, height)

        # Re-add ROI if exists
        if self.roi_path:
            pen = QPen(Qt.red, 2)
            brush = QBrush(Qt.red, Qt.Dense4Pattern)
            self.scene.addPath(self.roi_path, pen, brush)

    def eventFilter(self, source, event):
        if event.type() == event.MouseButtonPress and event.button() == Qt.RightButton:
            self.drawing = True
            self.path = QPainterPath()
            self.points = [self.view.mapToScene(event.pos())]
            self.path.moveTo(self.points[0])
            return True

        elif event.type() == event.MouseMove and self.drawing:
            pt = self.view.mapToScene(event.pos())
            self.path.lineTo(pt)
            self.points.append(pt)
            self.update_overlay_path(self.path)
            return True

        elif (
            event.type() == event.MouseButtonRelease
            and event.button() == Qt.RightButton
            and self.drawing
        ):
            self.drawing = False
            self.path.closeSubpath()
            self.roi_path = self.path
            self.update_overlay_path(self.path)
            return True

        return False

    def update_overlay_path(self, path):
        self.update_frame(self.current_frame_index)
        pen = QPen(Qt.red, 2)
        brush = QBrush(Qt.red, Qt.Dense4Pattern)
        self.scene.addPath(path, pen, brush)

    def check_ROI(self):

        n = self.start_spin.value() - 1
        frame1 = self.image_stack[n].copy()

        # Extract polygon from painter path
        polygon = [QPointF(p.x(), p.y()) for p in self.roi_path.toSubpathPolygons()[0]]
        points = np.array([[int(p.x()), int(p.y())] for p in polygon], dtype=np.int32)

        # Convert to 3-channel grayscale to allow colored outline
        img_color = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

        # Draw ROI outline (solid red line; thickness 2)
        cv2.polylines(
            img_color,
            [points],
            isClosed=True,
            color=(255, 0, 0),  # Red color in BGR
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Convert to QImage
        height, width, channels = img_color.shape
        bytes_per_line = channels * width
        qimg = QImage(
            img_color.data, width, height, bytes_per_line, QImage.Format_RGB888
        )

        # Show in popup dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("ROI Overlay")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(label)
        dialog.exec_()

        # n = self.start_spin.value() - 1
        # frame1 = self.image_stack[n]
        # mask = np.zeros_like(frame1, dtype=np.uint8)
        # polygon = [QPointF(p.x(), p.y()) for p in self.roi_path.toSubpathPolygons()[0]]
        # points = np.array([[int(p.x()), int(p.y())] for p in polygon])
        # cv2.fillPoly(mask, [points], 1)

        # # Show the masked image
        # masked_image = cv2.bitwise_and(frame1, frame1, mask=mask)
        # qimg = QImage(
        #     masked_image.data,
        #     masked_image.shape[1],
        #     masked_image.shape[0],
        #     QImage.Format_Grayscale8,
        # )

        # dialog = QDialog(self)
        # dialog.setWindowTitle("ROI Image")
        # layout = QVBoxLayout(dialog)
        # label = QLabel()
        # label.setPixmap(QPixmap.fromImage(qimg))
        # layout.addWidget(label)

        # dialog.exec_()

    def compute_std(self):
        if self.image_stack is None or self.roi_path is None:
            self.status_label.setText("Error: Load image and draw ROI first.")
            return

        if not self.use_roi_checkbox.isChecked():
            self.status_label.setText("Error: ROI is not used.")
            return

        n = self.start_spin.value()
        dn = self.dt_spin.value()

        if n + dn >= self.image_stack.shape[0]:
            self.status_label.setText("Error: Frame index out of range.")
            return

        self.std_all = []
        self.diff_sequence = []
        self.avg_std = None
        self.status_label.setText("Computing STD...")
        # Make ROI mask for calculation
        frame1 = self.image_stack[n - 1]
        mask = np.zeros_like(frame1, dtype=np.uint8)
        polygon = [QPointF(p.x(), p.y()) for p in self.roi_path.toSubpathPolygons()[0]]
        points = np.array([[int(p.x()), int(p.y())] for p in polygon])
        cv2.fillPoly(mask, [points], 1)

        for i in range(n - 1, self.image_stack.shape[0] - dn, dn):
            frame1 = self.image_stack[i]
            frame2 = self.image_stack[i + dn]
            diff = frame2.astype(np.float32) - frame1.astype(np.float32)
            self.std_all.append(np.std(diff[mask == 1]))
            self.diff_sequence.append(diff[mask == 1])

        all_diffs = np.concatenate(self.diff_sequence)
        std_val = np.std(all_diffs)
        self.avg_std = std_val
        # print(f"Subtraction STD: {std_all}") # Debugging line
        self.status_label.setText(f"Subtraction STD in all ROI: {std_val:.3f}")

    def check_std(self):
        if self.avg_std is None or not hasattr(self, "std_all") or not self.std_all:
            self.status_label.setText("Error: Compute STD first.")
            return

        self.trend_window = QMainWindow()
        self.trend_window.setWindowTitle("STD Trend")

        main_widget = QWidget()
        self.trend_window.setCentralWidget(main_widget)

        # Table widget
        table = QTableWidget()
        table.setRowCount(len(self.std_all))
        table.setColumnCount(1)
        table.setHorizontalHeaderLabels(["STD Value"])
        for i, val in enumerate(self.std_all):
            table.setItem(i, 0, QTableWidgetItem(f"{val:.3f}"))

        # Plot widget
        fig = Figure(figsize=(2, 1))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        x_vals = list(range(1, len(self.std_all) + 1))  # Start from 1
        ax.plot(x_vals, self.std_all, marker="o", linestyle="-", color="blue")
        ax.set_title("STD Trend")
        ax.set_xlabel("Index (starting from 1)")
        ax.set_ylabel("STD")
        ax.grid(True)

        # Average label
        avg_label = QLabel(f"Average STD: {self.avg_std:.3f} in {self.mode} mode")
        avg_label.setAlignment(Qt.AlignCenter)
        avg_label.setStyleSheet("font-weight: bold; margin-top: 12px;")

        # Layouts
        hbox = QHBoxLayout()
        hbox.addWidget(table, 1)
        hbox.addWidget(canvas, 2)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(avg_label)

        main_widget.setLayout(vbox)

        self.trend_window.resize(800, 400)
        self.trend_window.show()

    def use_result(self):
        if (
            self.avg_std is None
            or not hasattr(self, "std_all")
            or not hasattr(self, "diff_sequence")
        ):
            self.status_label.setText("Error: Compute STD first.")
            return
        if self.callback:
            self.callback(self.avg_std)  # pass result to main window

        self.status_label.setText(f"Using result: {self.avg_std:.3f}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdSetupWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
