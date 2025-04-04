'''
new RPOC GUI/model
goal is to have a fully single-"segment" level editor, both in table and drawing form
i want it to feel like the cellpose GUI, so I have changed to PyQT for this part
tkinter is more familiar, so I will keep that for the main GUIs
'''

import sys
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QCheckBox, QLabel, QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QBrush, QPainter
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint


class ImageViewer(QGraphicsView):
    def __init__(self, scene, roi_table):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.roi_table = roi_table

        self.drawing = False
        self.current_path = None
        self.temp_path_item = None
        self.current_points = []

        self.show_rois = True # from M click

        self.roi_items = [] 
        self.roi_count = 0

        self.path_pen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.roi_opacity = 0.4

        self.brush_radius = 3
        self.cursor_brush = QGraphicsEllipseItem(0, 0, self.brush_radius * 2, self.brush_radius * 2)
        self.cursor_brush.setBrush(QBrush(Qt.blue))
        self.cursor_brush.setPen(QPen(Qt.NoPen))
        self.cursor_brush.setZValue(1000)  # Always on top
        self.cursor_brush.setVisible(False)
        self.scene().addItem(self.cursor_brush)

        self.setMouseTracking(True)
        self.add_cursor_brush()

    def mousePressEvent(self, event):
        # the goal here is for a right click to start the drawing
        # then as the cursor is moved, the drawing is done
        # then when right click is pressed again, the drawing is done
        # i want it to be just like cellpose, but maybe better polygon-ing logic?
        if event.button() == Qt.RightButton:
            self.drawing = True
            self.cursor_brush.setVisible(True)
            self.current_points = []
            self.current_path = QPainterPath()

            start_pos = self.mapToScene(event.pos())
            self.current_path.moveTo(start_pos)
            self.current_points.append(start_pos)

            if self.temp_path_item:
                self.scene().removeItem(self.temp_path_item)
            self.temp_path_item = self.scene().addPath(self.current_path, self.path_pen)
            self.temp_path_item.setZValue(999)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        self.cursor_brush.setPos(scene_pos.x() - self.brush_radius, scene_pos.y() - self.brush_radius)

        if self.drawing and self.current_path is not None:
            if not self.point_in_existing_rois(scene_pos):
                self.current_path.lineTo(scene_pos)
                self.current_points.append(scene_pos)
                if self.temp_path_item:
                    self.temp_path_item.setPath(self.current_path)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self.drawing:
            self.drawing = False
            self.cursor_brush.setVisible(False)

            if len(self.current_points) > 2:
                self.current_path.closeSubpath()

            if self.temp_path_item:
                self.scene().removeItem(self.temp_path_item)
                self.temp_path_item = None

            self.roi_count += 1
            color = self.get_random_color()
            roi_item = self.scene().addPath(self.current_path)
            roi_item.setPen(QPen(color, 2))
            roi_item.setBrush(QBrush(color))
            roi_item.setOpacity(self.roi_opacity if self.show_rois else 0.0)

            self.roi_items.append(roi_item)
            self.add_roi_to_table(self.roi_count, self.current_points)

            self.current_path = None
            self.current_points = []

        super().mouseReleaseEvent(event)

    def add_cursor_brush(self):
        # DEFENSE DEFENSE DEFENSE bum bum bum
        if hasattr(self, 'cursor_brush') and self.cursor_brush in self.scene().items():
            self.scene().removeItem(self.cursor_brush)

        from PyQt5.QtGui import QColor
        self.brush_radius = 4
        self.cursor_brush = self.scene().addEllipse(
            -self.brush_radius, -self.brush_radius,
            2 * self.brush_radius, 2 * self.brush_radius,
            QPen(Qt.NoPen),
            QBrush(QColor(255, 0, 0, 100))
        )
        self.cursor_brush.setZValue(10)  # Always on top

    def point_in_existing_rois(self, point):
        for roi_item in self.roi_items:
            if roi_item.path().contains(point):
                return True
        return False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_M:
            self.show_rois = not self.show_rois
            self.update_roi_visibility()
        else:
            super().keyPressEvent(event)

    def update_roi_visibility(self):
        for roi in self.roi_items:
            roi.setOpacity(self.roi_opacity if self.show_rois else 0)

    def add_roi_to_table(self, idx, points):
        row = self.roi_table.rowCount()
        self.roi_table.insertRow(row)
        # ROI names, will be helpful for organelle-level work
        self.roi_table.setItem(row, 0, QTableWidgetItem(f'ROI {idx}'))
        
        # string dark magic
        coords_str = ', '.join([f'({p.x():.1f}, {p.y():.1f})' for p in points])
        self.roi_table.setItem(row, 1, QTableWidgetItem(coords_str))

    def get_random_color(self):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        from PyQt5.QtGui import QColor
        return QColor(r, g, b)


class MainWindow(QMainWindow):
    # TODO: dark mode
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cellpose-like ROI Editor')

        self.image_scene = QGraphicsScene()
        self.roi_table = QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(['ROI Name', 'Coordinates'])

        self.image_view = ImageViewer(self.image_scene, self.roi_table)

        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        self.help_label = QLabel('Press "M" to toggle mask visibility')

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

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_scene.clear()
            self.image_scene.addPixmap(pixmap)
            self.image_scene.setSceneRect(QRectF(pixmap.rect()))

            # avoid crash on cursor deletion
            self.image_view.add_cursor_brush()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
