from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QVBoxLayout,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QMenu,
    QInputDialog,
    QDialog,
    QGridLayout,
    QGroupBox,
    QGraphicsPixmapItem,
    QColorDialog,
    QLineEdit,
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
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QEvent, Qt, QTimer, QRunnable, QThread, QThreadPool, QObject, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyrpoc.mains import acquisition
from pyrpoc.helpers.galvo_funcs import Galvo
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd
from matplotlib import cm, colors
from matplotlib.colorbar import ColorbarBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import tifffile
import cv2
import os
import io
import sys
from PIL import Image, ImageDraw, ImageFont


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
            self, "Select File(s)", "", "TIFF (*.tiff);;Text (*.txt);;All Files (*)"
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
        elif first_file.endswith(".tiff"):  # Assume multiple TIFFs
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



class DrawROIWindow(QDialog):
    roi_added = pyqtSignal(int, str, tuple, list)  # (index, name, color, polygon points)

    def __init__(self, image, existing_rois,  parent = None):
        super().__init__(parent)
        self.setWindowTitle("Manual ROI Drawing")
        self.setModal(False)
        self.image = image
        self.existing_rois = existing_rois  # list of QPainterPath or polygons

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        self.scene.addItem(self.pixmap_item)

        self.view.viewport().installEventFilter(self)
        self.drawing = False
        self.points = []

        self.roi_items = []  # Store drawn ROI QGraphicsPathItem
        self.current_item = None  # Current picked ROI item

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            pos = self.view.mapToScene(event.pos())
            if event.button() == Qt.RightButton:
                self.drawing = True
                self.path = QPainterPath(pos)
                self.points = [pos]
                self.temp_item = None
                return True
            elif event.button() == Qt.LeftButton:
                # Check if user clicked on existing ROI
                for item in self.roi_items:
                    if item.path().contains(pos):
                        self.show_context_menu(pos, item)
                        return True

        elif event.type() == QEvent.MouseMove and self.drawing:
            pos = self.view.mapToScene(event.pos())
            self.path.lineTo(pos)
            self.points.append(pos)
            self.update_overlay(self.path)
            return True

        elif event.type() == QEvent.MouseButtonRelease and self.drawing:
            if event.button() == Qt.RightButton:
                self.drawing = False
                if not np.array_equal(self.points[0],self.points[-1]):
                   self.points.insert(0, self.points[-1])
                self.path.closeSubpath()
                self.update_overlay(self.path, finalize=True)
                return True

        return False

    def update_overlay(self, path, finalize=False):
        pen = QPen(Qt.red, 2)
        brush = QBrush(Qt.red, Qt.Dense4Pattern)
        if finalize:
            self.scene.removeItem(self.temp_item)  # Remove temporary item if exists
            item = self.scene.addPath(path, pen, brush)
            self.roi_items.append(item)
        else:
            if hasattr(self, 'temp_item') and self.temp_item:
            # Draw preview only (temporary path), remove previous preview if needed
                self.scene.removeItem(self.temp_item)
            self.temp_item = self.scene.addPath(path, pen, brush)


    def show_context_menu(self, pos, item):
        menu = QMenu()
        add_action = menu.addAction("Add ROI")
        del_action = menu.addAction("Delete ROI")
        action = menu.exec_(self.mapToGlobal(self.view.mapFromScene(pos)))


        if action == add_action:
            idx, ok1 = QInputDialog.getInt(self, "ROI Index", "Enter index:")
            if not ok1: return
            name, ok2 = QInputDialog.getText(self, "ROI Name", "Enter name:")
            if not ok2: return
            color = QColorDialog.getColor()
            if not color.isValid(): return

            rgb = (color.red(), color.green(), color.blue())
            points = [(int(p.x()), int(p.y())) for p in self.points]
            self.roi_added.emit(idx, name, rgb, points)

        elif action == del_action:
            self.scene.removeItem(item)
            self.roi_items.remove(item)
    


class RealTimeTrackingDialog(QDialog):
    def __init__(self, main_gui):   
        super().__init__()
        self.setWindowTitle("Real-Time Cell Viability Tracking")
        self.main_gui = main_gui
        self.canceled = False
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setFocusPolicy(Qt.StrongFocus)
        self.resize(800, 600)
        
        main_layout = QHBoxLayout(self)
        self.setLayout(main_layout)
        
        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.setContentsMargins(10, 10, 10, 10)
        self.sidebar_layout.setSpacing(10)
        
        self.save_group = QGroupBox("Save Options")
        self.save_group.setFixedWidth(300)
        save_layout = QGridLayout(self.save_group)
        
        save_layout.addWidget(QLabel("Save Directory:"), 0, 0)
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText("Select folder...")
        save_layout.addWidget(self.save_dir_edit, 0, 1)
        browse_btn = QPushButton("📂")
        browse_btn.clicked.connect(self.browse_save_folder)
        save_layout.addWidget(browse_btn, 0, 2)
        
        self.save_images_checkbox = QCheckBox("Save Images")
        self.save_images_checkbox.setChecked(True)
        save_layout.addWidget(self.save_images_checkbox, 1, 0, 1, 2)
        
        self.save_subtraction_checkbox = QCheckBox("Save Subtraction STD Result")
        self.save_subtraction_checkbox.setChecked(True)
        save_layout.addWidget(self.save_subtraction_checkbox, 2, 0, 1, 2)
        
        self.save_binary_ROIs_checkbox = QCheckBox("Save binary ROIs")
        self.save_binary_ROIs_checkbox.setChecked(True)
        save_layout.addWidget(self.save_binary_ROIs_checkbox, 3, 0, 1, 2)
    
        
        
        # Create ROI group
        self.create_ROIs_group = QGroupBox("Create ROIs")
        self.create_ROIs_group.setFixedWidth(300)
        create_roi_layout = QGridLayout(self.create_ROIs_group)
        
        self.create_ROIs_button = QPushButton("Manual Selection")
        self.create_ROIs_button.clicked.connect(self.launch_manual_draw)
        create_roi_layout.addWidget(self.create_ROIs_button, 0, 0, 1, 2)
        
        self.show_ROIs_info_list = QTableWidget(0, 4)
        self.show_ROIs_info_list.setHorizontalHeaderLabels(["ROI Index", "Name", "Pixel Count", "Color"])
        self.show_ROIs_info_list.setEditTriggers(QTableWidget.NoEditTriggers)
        create_roi_layout.addWidget(self.show_ROIs_info_list, 1, 0, 1, 2)
        
        self.preview_ROI_button = QPushButton("Preview ROIs")
        self.preview_ROI_button.clicked.connect(self.preview_ROIs_create)
        create_roi_layout.addWidget(self.preview_ROI_button, 2, 0, 1, 1)
        
        self.save_ROI_button = QPushButton("Save ROIs")
        self.save_ROI_button.clicked.connect(self.save_ROIs)
        create_roi_layout.addWidget(self.save_ROI_button, 2, 1, 1, 1)

        self.use_ROI_button = QPushButton("Use ROIs")
        self.use_ROI_button.clicked.connect(self.use_ROIs_in_create)
        create_roi_layout.addWidget(self.use_ROI_button, 3, 0, 1, 2)

        # Load ROIs group
        self.setting_up_rois = False
        self.load_ROIs_group = QGroupBox("Load ROIs")
        self.load_ROIs_group.setFixedWidth(300)
        load_roi_layout = QGridLayout(self.load_ROIs_group)
    
        self.load_ROIs_button = QPushButton("Load ROIs from File")
        self.load_ROIs_button.clicked.connect(self.load_ROIs_from_files)
        load_roi_layout.addWidget(self.load_ROIs_button, 0, 0, 1, 2)
        
        self.load_ROIs_info_list = QTableWidget(0, 4)
        self.load_ROIs_info_list.setHorizontalHeaderLabels(["ROI Index", "Name", "Pixel Count", "Color"])
        self.load_ROIs_info_list.cellChanged.connect(self.on_roi_color_changed)
        self.num_rois = 0  # Initialize number of ROIs
        self.num_rois_create = 0
        self.num_rois_used = 0
        self.roi_colors = {}  # Store colors for each ROI index
        self.roi_colors_create = {}
        load_roi_layout.addWidget(self.load_ROIs_info_list, 1, 0, 1, 2)
        
        self.load_ROIs_preview_button = QPushButton("Preview Loaded ROIs")
        self.load_ROIs_preview_button.clicked.connect(self.preview_ROIs)
        load_roi_layout.addWidget(self.load_ROIs_preview_button, 2, 0, 1, 2)

        self.use_ROI_load_button = QPushButton("Use ROIs")
        self.use_ROI_load_button.clicked.connect(self.use_ROIs_in_load)
        load_roi_layout.addWidget(self.use_ROI_load_button, 3, 0, 1, 2)
        
        
        
        # Real-time tracking group
        self.real_time_tracking_group = QGroupBox("Real-Time Tracking")
        self.real_time_tracking_group.setFixedWidth(300)
        real_time_layout = QVBoxLayout(self.real_time_tracking_group)
        self.auto_fill_threshold_button = QPushButton("Auto Fill Thresholds")
        self.auto_fill_threshold_button.clicked.connect(self.auto_fill_threshold)
        real_time_layout.addWidget(self.auto_fill_threshold_button)
        
        self.low_threshold_label = QLabel("Low Threshold:")
        self.low_threshold_spin = QDoubleSpinBox()
        self.low_threshold_spin.setRange(0, 50)
        self.low_threshold_spin.setValue(0)
        self.high_threshold_label = QLabel("High Threshold:")
        self.high_threshold_spin = QDoubleSpinBox()
        self.high_threshold_spin.setRange(0, 50)
        self.high_threshold_spin.setValue(10)
        real_time_layout.addWidget(self.low_threshold_label)
        real_time_layout.addWidget(self.low_threshold_spin)
        real_time_layout.addWidget(self.high_threshold_label)
        real_time_layout.addWidget(self.high_threshold_spin)
        
        self.low_threshold_value = 0
        self.high_threshold_value = 0
        self.drop_percentage = 0
        self.roi_std_all = []
        
        # Initialize ROI mask
        self.roi_mask = np.zeros_like(self.main_gui.data[0], dtype=np.uint8) if self.main_gui.data else None
        self.roi_mask_created = np.zeros_like(self.main_gui.data[0], dtype=np.uint8) if self.main_gui.data else None
        self.roi_mask_used = np.zeros_like(self.main_gui.data[0], dtype=np.uint8) if self.main_gui.data else None
        
        # Tracking setup
        self.tracking_delta_frames_label = QLabel("Tracking Δn (frames):")
        self.tracking_delta_frames_spin = QSpinBox()
        self.tracking_delta_frames_spin.setRange(1, 50)
        self.tracking_delta_frames_spin.setValue(5)
        real_time_layout.addWidget(self.tracking_delta_frames_label)
        real_time_layout.addWidget(self.tracking_delta_frames_spin)
        
        self.start_tracking_button = QPushButton("Start Tracking")
        self.start_tracking_button.clicked.connect(self.prepare_run)
        self.stop_tracking_button = QPushButton("Stop Tracking")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)
        self.stop_tracking_button.setEnabled(False)
        real_time_layout.addWidget(self.start_tracking_button)
        real_time_layout.addWidget(self.stop_tracking_button)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFixedWidth(300)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        real_time_layout.addWidget(self.status_label)
        
        self.sidebar_layout.addWidget(self.save_group)
        self.sidebar_layout.addWidget(self.create_ROIs_group)
        self.sidebar_layout.addWidget(self.load_ROIs_group)
        self.sidebar_layout.addWidget(self.real_time_tracking_group)
        main_layout.addLayout(self.sidebar_layout)
        
        ## Rightside display
        # Graphics View
        self.display_scene = QGraphicsScene(self)
        self.display_view = QGraphicsView(self.display_scene)
        self.display_view.setFixedWidth(600)
        self.display_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        
        self.display_pixmap_item = QGraphicsPixmapItem()
        self.display_scene.addItem(self.display_pixmap_item)

        # Colorbar
        self.colorbar_widget = QWidget()
        self.colorbar_widget.setFixedWidth(80)  # Wider to accommodate labels
        self.colorbar_widget.setMinimumHeight(300) 
        
        self.colorbar_layout = QVBoxLayout(self.colorbar_widget)
        self.colorbar_layout.setContentsMargins(0, 0, 0, 0)
        # Create initial empty colorbar
        self.colorbar_layout.addStretch(1)
        self.colorbar_fig = Figure(figsize=(0.8, 4))
        self.colorbar_canvas = FigureCanvas(self.colorbar_fig)
        self.colorbar_layout.addWidget(self.colorbar_canvas)
        self.colorbar_layout.addStretch(1)

        # Display Layout
        display_layout = QHBoxLayout()
        display_layout.addWidget(self.display_view)
        display_layout.addWidget(self.colorbar_widget, alignment=Qt.AlignRight)

        main_layout.addLayout(display_layout)
        main_layout.setStretch(0,1)
        main_layout.setStretch(1,2)
        self.setLayout(main_layout)
        
        
        
        

    @pyqtSlot(int, int)
    def on_roi_color_changed(self, row, column):
        if self.setting_up_rois:  # Prevent triggering during setup
            return 
        if column != 3:
            return

        item = self.load_ROIs_info_list.item(row, column)
        if not item:
            return

        color_text = item.text()
        try:
            # Convert string to tuple
            r, g, b = eval(color_text.strip())
            assert all(0 <= v <= 255 for v in (r, g, b))
        except:
            self.status_label.setText(f"Invalid RGB at row {row}.")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            return
        self.roi_colors[row] = (r, g, b)  # update internal color map
        item.setBackground(QColor(r, g, b))


    def prepare_roi_colors(self, num_rois):
        for i in range(num_rois):
            color_item = self.load_ROIs_info_list.item(i, 3)
            if color_item:
                try:
                    color_tuple = eval(color_item.text())  # e.g., "(255,0,0)"
                    # check if in valid range, otherwise fallback to red
                    assert len(color_tuple) == 3 and all(0 <= v <= 255 for v in color_tuple)
                    self.roi_colors[i+1] = color_tuple
                except:
                    self.roi_colors[i+1] = (255, 0, 0)  # fallback color
        return 
            
    def show_colored_roi_preview(self, mask, roi_colors):
        """
        mask: 2D np.array with ROI index (0 = background, 1 = CELL1, 2 = CELL2...)
        roi_colors: dict mapping index (int) -> (R, G, B) tuple
        """
        h, w = mask.shape
        color_img = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA

        for roi_idx, color in roi_colors.items():
            if roi_idx == 0:  # background
                continue
            mask_roi = mask == roi_idx
            color_img[mask_roi, :3] = color  # RGB
            color_img[mask_roi, 3] = 120     # Alpha (transparency)

        qimg = QImage(color_img.data, w, h, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)

        dialog = QDialog(self)
        dialog.setWindowTitle("ROI Colored Mask Preview")
        layout = QVBoxLayout(dialog)
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)

        dialog.resize(min(w, 1000), min(h, 800))
        dialog.exec_()
    
    def browse_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder: self.save_dir_edit.setText(folder)
    
    def handle_new_roi(self, index, name, color, points):
        self.num_rois_create += 1
        row = self.show_ROIs_info_list.rowCount()
        self.show_ROIs_info_list.insertRow(row)

        self.show_ROIs_info_list.setItem(row, 0, QTableWidgetItem(str(index)))
        self.show_ROIs_info_list.setItem(row, 1, QTableWidgetItem(name))

        mask = np.zeros_like(self.main_gui.data[0], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
        pixel_count = int(mask.sum())
        self.show_ROIs_info_list.setItem(row, 2, QTableWidgetItem(str(pixel_count)))
        self.show_ROIs_info_list.setItem(row, 3, QTableWidgetItem(str(color)))
        # Create color picker
        # Create color picker
        color_item = self.show_ROIs_info_list.item(row,3)
        color_item.setBackground(QColor(*color))  

        self.roi_colors_create[index] = color
        self.roi_mask_created[mask == 1] = index

    def launch_manual_draw(self):
        # clear the ROI info region
        self.show_ROIs_info_list.setRowCount(0)
        self.roi_mask_created = np.zeros_like(self.main_gui.data[0], dtype=np.uint8) if self.main_gui.data else None
        # We only use the first channel (EB3 channel) ????????
        current_frame = self.main_gui.data[0]
        if current_frame is None:
            self.status_label.setText("Error: No image in current channel.")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            return
        img = current_frame.astype(np.uint8)
        # rescale to 255 ######## ?? can be changed later about scaling
        img = img*255
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()

        self.drawing_dialog = DrawROIWindow(qimg, existing_rois=[], parent=self)
        self.drawing_dialog.roi_added.connect(self.handle_new_roi)
        self.drawing_dialog.exec_()
    
    def auto_fill_threshold(self):
        mode = self.main_gui.threshold_mode.get()
        if mode == 1:
            try:
                self.low_threshold_value = float(self.main_gui.predefined_low_entry.get())
                self.high_threshold_value = float(self.main_gui.predefined_high_entry.get())
                self.low_threshold_spin.setValue(self.low_threshold_value)
                self.high_threshold_spin.setValue(self.high_threshold_value)
                self.status_label.setText("Using: Predefined threshold.")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
            except ValueError:
                self.status_label.setText("Error:" + str(e))
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
        elif mode == 2:
            try:
                self.drop_percentage = float(self.main_gui.percent_drop_entry.get())
                self.low_threshold_spin.setValue(0)
                self.high_threshold_spin.setValue(0)
                self.status_label.setText("Using: Percentage-drop threshold, initiate with 0, awaiting to start...")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
            except ValueError as e:
                self.status_label.setText("Error:" + str(e))
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
        elif mode == 3:
            try:
                self.high_threshold_value = self.live_avg
                self.low_threshold_value = self.dead_avg
                self.low_threshold_spin.setValue(self.low_threshold_value)
                self.high_threshold_spin.setValue(self.high_threshold_value)
                self.status_label.setText("Using: Experiment-based threshold.")
                self.status_label.setStyleSheet("font-weight: bold; color: green;")
            except ValueError:
                self.status_label.setText("Error:" + str(e))
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
        else:
            return
        

    def load_ROIs_from_files(self):
        # all files allowed
        file_path, _ = QFileDialog.getOpenFileNames(
            self, "Select ROI File(s)", "", " All Files (*)"
        )
        # ------------------1. Check selected files number and file type and file content-------------------
        if not file_path:
            return
        if len(file_path) != 2:
            self.status_label.setText("Please select two files(.txt and .tiff) for one ROI.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        
        txt_file = file_path[0].lower()
        tiff_file = file_path[1].lower()
        if txt_file.endswith(".txt") and tiff_file.endswith(".tiff"):
            self.roi_info_path = txt_file
            self.roi_mask_path = tiff_file
            self.roi_mask = tifffile.imread(tiff_file)
        elif tiff_file.endswith(".txt") and txt_file.endswith(".tiff"):
            txt_file = file_path[1].lower()
            tiff_file = file_path[0].lower()
            self.roi_info_path = txt_file
            self.roi_mask_path = tiff_file
            self.roi_mask = tifffile.imread(tiff_file)
        else:
            self.status_label.setText("Please select a valid text file and a TIFF file.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        
        # ------------------2. Load ROI info into table and count pixels of each ROIs-------------------
        try:
            # demo of .txt file content:
            # 3
            # 0,CELL1
            # 1,CELL2
            # 2,CELL3
            with open(self.roi_info_path, 'r') as f:
                lines = f.readlines()
            if not lines:
                self.status_label.setText("Error: .txt file is empty.")
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
                return
            self.num_rois = int(lines[0].strip()) # First line should contain the number of ROIs
            if self.num_rois != len(lines) - 1: 
                self.status_label.setText("Error: Number of ROIs does not match the file content.")
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
                return
            self.load_ROIs_info_list.setRowCount(self.num_rois)
            
            self.setting_up_rois = True
            
            for i, line in enumerate(lines[1:self.num_rois + 1]): # Lines after the first line contain ROI info
                index, name = line.strip().split(',')
                self.load_ROIs_info_list.setItem(i, 0, QTableWidgetItem(index))
                self.load_ROIs_info_list.setItem(i, 1, QTableWidgetItem(name))
            
            # Load the ROI mask
            try:
                self.roi_mask = cv2.imread(self.roi_mask_path, cv2.IMREAD_GRAYSCALE)
                self.roi_mask = np.array(self.roi_mask)
                if self.roi_mask is None or self.roi_mask.size !=2:
                    Exception("Invalid ROI mask image.")
                counts = np.bincount(self.roi_mask.ravel())
                for i in range(self.num_rois):
                    # Create pixel count item
                    pixel_count = counts[i+1] 
                    self.load_ROIs_info_list.setItem(i, 2, QTableWidgetItem(str(pixel_count)))
                    # Create color picker
                    rgb = (255, 0, 0)  # Example RGB tuple for the ROI
                    color_text = str(rgb)  # Converts to "(255, 0, 0)"
                    color_item = QTableWidgetItem(color_text)
                    self.load_ROIs_info_list.setItem(i, 3, color_item)
                    color_item.setBackground(QColor(*rgb))  
                self.setting_up_rois = False

            except Exception as e:
                self.status_label.setText(f"Error loading ROIs: {str(e)}")
                self.status_label.setStyleSheet("font-weight: bold; color: red;")
                return
                
        except Exception as e:
            self.status_label.setText(f"Error reading ROI info file: {str(e)}")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            return
    
    def save_ROIs(self):

        if self.save_dir_edit.text() == "":
            self.status_label.setText(f"Choose save dir first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        else:
            os.makedirs(self.save_dir_edit.text(),exist_ok=True)       
        try:
            # save tiff
            tiff_path = os.path.join(self.save_dir_edit.text(), "ROI_mask.tiff")
            tifffile.imwrite(tiff_path, self.roi_mask_created.astype(np.uint8))
            
            # save txt
            txt_path = os.path.join(self.save_dir_edit.text(), "ROI_info.txt")
            with open(txt_path, "w") as f:
                f.write(f"{self.num_rois_create}\n")
                for row in range(self.num_rois_create):
                    index_item =  self.show_ROIs_info_list.item(row,0)
                    name_item = self.show_ROIs_info_list.item(row,1)
                    if index_item and name_item:
                        index = index_item.text()
                        name = name_item.text()
                        f.write(f"{index},{name}\n")
            
            self.status_label.setText(f"Saved ROIs mask and info.")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        except Exception as e:
            self.status_label.setText(f"Error saving ROIs mask and info.")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
    
        return
    
    def use_ROIs_in_create(self):
        not_all_zero_load = np.any(self.roi_mask)
        not_all_zero_create = np.any(self.roi_mask_created)
        if self.roi_mask_created is None or not not_all_zero_create:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        elif not_all_zero_load:
            self.roi_mask_used = self.roi_mask_created
            self.num_rois_used = self.num_rois_create
            self.status_label.setText(f"Warning: overwrite ROIs with created ones...")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.roi_mask_used = self.roi_mask_created
            self.num_rois_used = self.num_rois_create
            self.status_label.setText(f"Using created ROIs.")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        
    def use_ROIs_in_load(self):
        not_all_zero_create = np.any(self.roi_mask_created)
        not_all_zero_load = np.any(self.roi_mask)
        if self.roi_mask is None or not not_all_zero_load:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        elif not_all_zero_create:
            self.roi_mask_used = self.roi_mask
            self.num_rois_used = self.num_rois
            self.status_label.setText(f"Warning: overwrite ROIs with loaded ones...")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.roi_mask_used = self.roi_mask
            self.num_rois_used = self.num_rois
            self.status_label.setText(f"Using loaded ROIs.")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")

    def preview_ROIs_create(self):
        not_all_zero = np.any(self.roi_mask_created)
        if self.roi_mask_created is None:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        elif not_all_zero:
            self.prepare_roi_colors(self.num_rois_create)
            self.show_colored_roi_preview(self.roi_mask_created, self.roi_colors_create)   
        else:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
    
    def preview_ROIs(self):
        not_all_zero = np.any(self.roi_mask)
        if self.roi_mask is None:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        elif not_all_zero:
            self.prepare_roi_colors(self.num_rois)
            self.show_colored_roi_preview(self.roi_mask, self.roi_colors)
        else:
            self.status_label.setText("Notice: Get ROIs first.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
            return
        
    def update_status(self,text):
        QTimer.singleShot(0, lambda: self.status_label.setText(text))
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
    
    def prepare_run(self):
        self.low_threshold = self.low_threshold_spin.value()
        self.high_threshold = self.high_threshold_spin.value()
        self.update_colorbar(self.high_threshold, self.low_threshold)
        self.roi_std_all = [[] for _ in range(self.num_rois_used)]  
        
        # Read and store checkbox states
        self.save_data = self.save_images_checkbox.isChecked()
        self.save_subtraction = self.save_subtraction_checkbox.isChecked()
        self.save_binary_ROIs = self.save_binary_ROIs_checkbox.isChecked()
        
        if not self.save_data and not self.save_subtraction and not self.save_binary_ROIs:
            self.status_label.setText("Warning: Not saving anything.")
            self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
        else:
            if self.save_dir_edit.text() == "":
                self.status_label.setText(f"Warning: Choose save dir first or uncheck savebox.")
                self.status_label.setStyleSheet("font-weight: bold; color: yellow;")
                return
            
        self.worker = RealTimeTrackingWorker(self.main_gui, self.tracking_delta_frames_spin.value())
        self.worker.error.connect(lambda msg: self.update_status(f"Viability Tracking Error: {msg}"))
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(self.on_tracking_finished)
        self.worker.frame_ready.connect(self.on_frame_ready)
        
        self.start_tracking_button.setEnabled(False)
        self.stop_tracking_button.setEnabled(True)
        
        self.worker.start()
        self.update_status("Starting Real-time Viability Tracking...")
    
    def stop_tracking(self):
        self.status_label.setText("Status: Stopped.")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.running = False
            self.worker.wait(500)
        self.start_tracking_button.setEnabled(True)
        self.stop_tracking_button.setEnabled(False)
        
        return

    def update_display_image(self, raw_frame, overlay_array):
        """
        Combines grayscale image and RGBA overlay, then updates the display pixmap.
        """
        # Normalize grayscale frame 0-1 to 0–255
        gray_img = (raw_frame * 255).astype(np.uint8)
        gray_rgb = np.stack([gray_img] * 3, axis=-1)  # Convert to RGB shape (H, W, 3)

        # Blend overlay
        if overlay_array is not None:  # RGBA
            alpha = overlay_array[..., 3:4] / 255.0  # Normalize alpha
            blended = gray_rgb * (1 - alpha) + overlay_array[..., :3] * alpha
            blended = blended.astype(np.uint8)
        else:
            blended = gray_rgb  # fallback

        # Convert to QImage
        h, w, _ = blended.shape
        qimage = QImage(blended.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.display_pixmap_item.setPixmap(pixmap)
        self.display_pixmap_item.setOffset(0, 0)

    def update_colorbar(self, high_thresh, low_thresh):
        """Update colorbar with dark theme"""
        # Calculate range with mercy margin
        mercy_margin = 0.3 * (high_thresh - low_thresh)
        vmin = max(0, low_thresh - mercy_margin)  # Don't go below 0
        vmax = high_thresh + mercy_margin
        
        # Clear previous figure
        self.colorbar_fig.clear()
        
        # Set dark background for the figure
        self.colorbar_fig.patch.set_facecolor('#2e2e2e')  # Dark gray background
        
        # Create new colorbar
        ax = self.colorbar_fig.add_axes([0.05, 0.2, 0.05, 0.5])  # [left, bottom, width, height]
        
        # Set dark theme for the axes
        ax.set_facecolor('#2e2e2e')  # Match figure background
        ax.tick_params(axis='both', colors='white')  # White tick labels
        for spine in ax.spines.values():
            spine.set_edgecolor('white')  # White borders
        
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('RdYlGn')
        cmap.set_extremes(under = 'darkred', over = "darkgreen" )
        cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
        
        # Set ticks and labels with white color
        num_ticks = 5
        tick_values = np.linspace(vmin, vmax, num_ticks)
        tick_labels = [f"{val:.2f}" for val in tick_values]
        cb.set_ticks(tick_values)
        cb.set_ticklabels(tick_labels)
        cb.set_label('STD Value', rotation=270, labelpad=15, color='white')
        
        # Set colorbar label and ticks to white
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
        
        # Redraw
        self.colorbar_canvas.draw()
        
        # mercy_margin = 0.3 * (high_thresh - low_thresh)
        # self.color_min = low_thresh - mercy_margin
        # self.color_max = high_thresh + mercy_margin

        # # Create unnormalized range (actual std values)
        # raw_values = np.linspace(self.color_min, self.color_max, 256)

        # # Normalize to [0,1] for colormap
        # norm = (raw_values - self.color_min) / (self.color_max - self.color_min)
        # norm = np.clip(norm, 0, 1)  # safety

        # # Get RGBA values
        # rgba = (cm.jet(norm)[:, :3] * 255).astype(np.uint8)  # (256, 3)

        # # Stack to make a vertical color bar
        # width = 20
        # bar = np.tile(rgba[:, np.newaxis, :], (1, width, 1))  # (256, 20, 3)
        # bar = np.ascontiguousarray(bar)

        # # Convert to QImage and show
        # img = QImage(bar.data, width, 256, width * 3, QImage.Format_RGB888)
        # pixmap = QPixmap.fromImage(img)
        # self.colorbar_widget.setPixmap(pixmap)


    def _add_std_text(self, text_overlay, mask, std_value):
        """Add orange text label at ROI centroid using PIL"""
        # Find centroid coordinates
        y, x = np.where(mask)
        if len(x) == 0:
            return
            
        cx, cy = int(np.mean(x)), int(np.mean(y))
        
        # Create PIL image and drawing context
        img_pil = Image.fromarray(text_overlay)
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load font (fallback to default if needed)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("LiberationSans-Regular.ttf", 14)
            except:
                font = ImageFont.load_default()
        
        # Format text and calculate dimensions
        text = f"{std_value:.2f}"
        text_width = draw.textlength(text, font=font)  # Returns single float value
        text_height = font.size  # Get approximate height from font
        
        # Calculate position (centered)
        text_x = cx - text_width//2
        text_y = cy - text_height//2
        
        # Draw orange text (RGB: 255,165,0)
        draw.text((text_x, text_y), text, font=font, fill=(255, 165, 0, 255))
        
        # Update the numpy array
        text_overlay[:] = np.array(img_pil)
        
    # alpha is the transparency of the ROI on images
    def generate_overlay_from_std(self, roi_mask, subtracted_frame, low_thresh, high_thresh, colormap=cm.jet, alpha=0.5):
        """
        Generate RGBA overlay from ROI mask and subtracted image using STD and colormap.
        """
        h, w = roi_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        text_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # set mercy margin to accept out of threshold value
        mercy_margin = 0.2 *(high_thresh - low_thresh)
        color_min = max(0,low_thresh - mercy_margin)
        color_max = high_thresh + mercy_margin
        
        dark_red = np.array([139,0,0])
        dark_green = np.array([0, 100, 0])
        for roi_id in np.unique(roi_mask):
            if roi_id == 0:
                continue
            mask = roi_mask == roi_id
            roi_std = np.std(subtracted_frame[mask])
            self.roi_std_all[roi_id - 1].append(roi_std)  # Store std for this ROI
            
            # check if it's over/above high/low threshold
            if roi_std > color_max:
                overlay[mask] = [*dark_green, int(alpha * 255)]
            elif roi_std < color_min:
                overlay[mask] = [*dark_red, int(alpha * 255)]
            else:
                # Within range - use colormap
                norm = (roi_std - color_min) / (color_max - color_min)
                norm = np.clip(norm, 0, 1)  # Safety clamp
                cmap = plt.get_cmap('RdYlGn')
                r, g, b, _ = cmap(norm)
                overlay[mask] = [int(r * 255), int(g * 255), int(b * 255), int(alpha * 255)]
            self._add_std_text(text_overlay, mask, roi_std)
        # Combine overlays (text on top)
        text_mask = text_overlay[..., 3] > 0
        overlay[text_mask] = text_overlay[text_mask]
        return overlay
    
    
    @pyqtSlot(object, object)
    def on_frame_ready(self, subtracted_frame, data):
        # create Qimage, canvas
        if data is None:
            return
        # adding for save images
        if self.save_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(self.save_dir_edit.text(), f"{timestamp}-original_data.txt")
            with open(save_path, "a") as f:
                np.savetxt(f, data, fmt="%.6f",delimiter="\t")

        if subtracted_frame is None:
            # just show the original data
            self.update_display_image(data, None)
        else:  
            overlay_array = self.generate_overlay_from_std(self.roi_mask_used, subtracted_frame, self.low_threshold, self.high_threshold, alpha = 0.5)
            self.update_display_image(data, overlay_array)
        
        
    
    def save_roi_std_to_file(self, save_path):
        """
        Save self.roi_std_all to CSV with each ROI as a column.
        Each column header is 'ROI_{id}'.
        """
        # Pad lists to equal length
        max_len = max(len(lst) for lst in self.roi_std_all)
        padded_data = [lst + [np.nan] * (max_len - len(lst)) for lst in self.roi_std_all]

        # Create DataFrame
        df = pd.DataFrame(
            {f"ROI_{i+1}": padded_data[i] for i in range(len(padded_data))}
        )

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, float_format="%.6f")  # or .txt if preferred
    
    def save_ROIs_to_binary_mask(self, save_path):
        """
        Save the binary mask where any non-zero value in self.roi_mask_used becomes 255 (ROI),
        and 0 stays as background.
        """
        if self.roi_mask_used is None:
            print("No ROI mask to save.")
            return

        # save_path = os.path.join(self.save_dir_edit.text(), "binary_mask_all_ROIs.tif")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        binary_mask = (self.roi_mask_used != 0).astype(np.uint8) * 255  # ROI pixels → 255, background → 0
        img = Image.fromarray(binary_mask)
        img.save(save_path)

        
    @pyqtSlot()
    def on_tracking_finished(self):
        self.update_status("Viability Tracking finished.")
        print("tracking_finished_emit_Check")

        # Create save directory
        save_dir = self.save_dir_edit.text().strip()
        if not save_dir:
            self.update_status("Finished. No save directory selected. Skipping save.")
            return

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
        # Save ROI std data
        if self.save_subtraction:
            std_data_path = os.path.join(save_dir, f"{timestamp}-roi_std_data.csv")
            self.save_roi_std_to_file(std_data_path)
            self.update_status(f"Saved ROI STD data to {std_data_path}")

        # Save ROI binary image
        if self.save_binary_ROIs:
            ROIs_binary_mask_path =  os.path.join(save_dir, f"{timestamp}-roi_binary_mask.tiff")
            self.save_ROIs_to_binary_mask(ROIs_binary_mask_path)


        # Reset UI
        self.start_tracking_button.setEnabled(True)
        self.stop_tracking_button.setEnabled(False)
        
        
        
        
        
        
class RealTimeTrackingWorker(QThread):
    error = pyqtSignal(str)
    finished = pyqtSignal()
    status_update = pyqtSignal(str)
    frame_ready = pyqtSignal(object, object)
    
    def __init__(self, gui, delta_frames):
        super().__init__()
        self.gui = gui
        self.delta_frames = delta_frames
        self.running = True
        self.frames = [] # counter and container for subtraction
        self.frame_counter = 0
        
    def run(self):
        if not self.running: # just in case???
            return
        try:
            self.status_update.emit(f"Acquiring..")
            while self.running:
                
                # Prepare for acquire_single
                self.gui.update_config()
                galvo = Galvo(self.gui.config)
                channels = [f"{self.gui.config['device']}/{ch}" for ch in self.gui.config['ai_chans']]
                
                
                acquisition.acquire_single(self.gui, channels, galvo)
                data = getattr(self.gui, 'data', []) or []
                frame = data[0].astype(np.float32)
                self.frame_counter +=1
                
                self.frames.append(frame)
                subtracted_frame = None
                if len(self.frames) > self.delta_frames:
                    frame_t = self.frames[-self.delta_frames - 1]
                    frame_now = self.frames[-1]
                    subtracted_frame = frame_now - frame_t
                    self.status_update.emit(f"Subtraction based on frame:{self.frame_counter} and frame {self.frame_counter -self.delta_frames}")
                self.frame_ready.emit(subtracted_frame,frame)
                
        
                # control length of buffer to avoid overloading memory
                if len(self.frames)>100:
                    self.frames.pop(0)
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThresholdSetupWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
