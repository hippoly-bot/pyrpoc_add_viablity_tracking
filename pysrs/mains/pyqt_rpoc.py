'''
new RPOC GUI/model
goal is to have a fully single-"segment" level editor, both in table and drawing form
i want it to feel like the cellpose GUI, so I have changed to PyQT for this part
tkinter is more familiar, so I will keep that for the main GUIs
'''

import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QCheckBox, QLabel, QMenu, QGraphicsTextItem, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QBrush, QPainter, QFont, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QVariant, QPoint


class ImageViewer(QGraphicsView):
    def __init__(self, scene, roi_table):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.roi_table = roi_table

        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.live_path_item = None

        self._zoom = 0
        self._empty = True
        self._scene = scene

        self.drawing = False
        self.current_path = None
        self.current_points = []
        self.show_rois = True
        self.show_labels = True  # toggled by N
        self.roi_items = []      # list of QGraphicsPathItem
        self.roi_label_items = []# parallel list of QGraphicsTextItem
        self.path_pen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.roi_opacity = 0.4

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self._zoom += 1
        else:
            zoom_factor = zoom_out_factor
            self._zoom -= 1

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            if not self.drawing:
                self.drawing = True
                self.current_points = []
                self.current_path = QPainterPath()
                start_pos = self.mapToScene(event.pos())
                self.current_path.moveTo(start_pos)
                self.current_points.append(start_pos)

                if self.live_path_item:
                    self.scene().removeItem(self.live_path_item)
                self.live_path_item = self.scene().addPath(self.current_path, self.path_pen)
            else:
                self.drawing = False
                end_pos = self.mapToScene(event.pos())
                self.current_points.append(end_pos)
                self.current_path.lineTo(end_pos)
                self.current_path.closeSubpath()

                self.scene().removeItem(self.live_path_item)
                self.live_path_item = None

                new_index = len(self.roi_items) + 1
                roi_item = self.scene().addPath(self.current_path)
                color = self.get_random_color()
                roi_item.setPen(QPen(color, 2))
                roi_item.setBrush(QBrush(color))
                roi_item.setOpacity(self.roi_opacity if self.show_rois else 0.0)

                self.roi_items.append(roi_item)

                roi_label = self.create_roi_label(new_index, self.current_points)
                self.roi_label_items.append(roi_label)

                self.add_roi_to_table(new_index, self.current_points)

                # reset path
                self.current_path = None
                self.current_points = []

        elif event.button() == Qt.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and not self.drawing:
            self.setDragMode(QGraphicsView.NoDrag)
            self.viewport().setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def is_inside_any_roi(self, point):
        for roi_item in self.roi_items:
            if roi_item.contains(point):
                return True
        return False

    def find_boundary_point(self, p1, p2):
        steps = 50
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()

        last_outside = p1
        for i in range(1, steps+1):
            alpha = i / steps
            x = x1 + alpha * (x2 - x1)
            y = y1 + alpha * (y2 - y1)
            candidate = QPointF(x, y)
            if self.is_inside_any_roi(candidate):
                return last_outside
            last_outside = candidate
        return p2

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path is not None:
            new_pos = self.mapToScene(event.pos())

            if not self.current_points:
                self.current_points.append(new_pos)
                self.current_path.lineTo(new_pos)
            else:
                last_valid = self.current_points[-1]
                if self.is_inside_any_roi(new_pos):
                    boundary = self.find_boundary_point(last_valid, new_pos)
                    self.current_points.append(boundary)
                    self.current_path.lineTo(boundary)
                else:
                    self.current_points.append(new_pos)
                    self.current_path.lineTo(new_pos)

            if self.live_path_item:
                self.live_path_item.setPath(self.current_path)

        super().mouseMoveEvent(event)

    def create_roi_label(self, roi_index, points):
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)

        label_text = f"{roi_index}"
        label_item = QGraphicsTextItem(label_text)
        
        font = QFont("Arial", 14, QFont.Bold)
        label_item.setFont(font)
        label_item.setDefaultTextColor(Qt.white)

        text_rect = label_item.boundingRect()
        label_item.setPos(cx - text_rect.width() / 2, cy - text_rect.height() / 2)
        label_item.setZValue(999)

        label_item.setVisible(self.show_rois and self.show_labels)
        self.scene().addItem(label_item)
        return label_item



    def keyPressEvent(self, event):
        if event.key() == Qt.Key_M:
            self.show_rois = not self.show_rois
            self.update_roi_visibility()
        elif event.key() == Qt.Key_N:
            self.show_labels = not self.show_labels
            self.update_roi_visibility()
        else:
            super().keyPressEvent(event)

    def update_roi_visibility(self):
        for roi_item in self.roi_items:
            roi_item.setOpacity(self.roi_opacity if self.show_rois else 0)
        for lbl in self.roi_label_items:
            lbl.setVisible(self.show_rois and self.show_labels)

    def add_roi_to_table(self, idx, points):
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)
        item = QTableWidgetItem(f'ROI {idx}')
        item.setData(Qt.UserRole, idx)
        self.roi_table.setItem(row, 0, item)

        coords_str = ', '.join([f'({p.x():.1f}, {p.y():.1f})' for p in points])
        self.roi_table.setItem(row, 1, QTableWidgetItem(coords_str))

    def get_random_color(self):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        from PyQt5.QtGui import QColor
        return QColor(r, g, b)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('New RPOC Editor')

        self.image_scene = QGraphicsScene()
        self.roi_table = QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(['ROI Name', 'Coordinates'])

        self.image_view = ImageViewer(self.image_scene, self.roi_table)

        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        self.help_label = QLabel('Press \"M\" to toggle mask visibility\n'
                                 'Press \"N\" to toggle label visibility\n'
                                 'Right-click row in table to delete ROI')

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(load_button)
        left_layout.addWidget(self.help_label)
        left_layout.addWidget(self.image_view)

        layout.addLayout(left_layout)
        layout.addWidget(self.roi_table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self.show_table_context_menu)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_scene.clear()
            self.image_scene.addPixmap(pixmap)
            self.image_scene.setSceneRect(QRectF(pixmap.rect()))

    def show_table_context_menu(self, pos):
        # check which row was clicked
        row = self.roi_table.indexAt(pos).row()
        if row < 0:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Delete ROI")
        action = menu.exec_(self.roi_table.mapToGlobal(pos))
        if action == delete_action:
            self.delete_roi_row(row)

    def delete_roi_row(self, row):
        idx_item = self.roi_table.item(row, 0)
        if not idx_item:
            return
        roi_idx = idx_item.data(Qt.UserRole)
        if roi_idx is None:
            return

        self.roi_table.removeRow(row)
        viewer = self.image_view

        label_to_remove = None
        roi_to_remove = None
        remove_i = None
        for i, lbl in enumerate(viewer.roi_label_items):
            if lbl.toPlainText() == str(roi_idx):
                label_to_remove = lbl
                roi_to_remove = viewer.roi_items[i]
                remove_i = i
                break

        if remove_i is not None:
            viewer.scene().removeItem(roi_to_remove)
            viewer.scene().removeItem(label_to_remove)
            viewer.roi_items.pop(remove_i)
            viewer.roi_label_items.pop(remove_i)

        for i, (roi, label) in enumerate(zip(viewer.roi_items, viewer.roi_label_items)):
            idx = i + 1  
            label.setPlainText(str(idx))

            path = roi.path()
            poly = path.toSubpathPolygons()[0]
            cx = sum(p.x() for p in poly) / len(poly)
            cy = sum(p.y() for p in poly) / len(poly)
            text_rect = label.boundingRect()
            label.setPos(cx - text_rect.width() / 2, cy - text_rect.height() / 2)

        viewer.roi_table.setRowCount(0)
        for i, roi in enumerate(viewer.roi_items):
            path = roi.path()
            points = path.toSubpathPolygons()[0]
            viewer.add_roi_to_table(i + 1, points)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
