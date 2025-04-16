import numpy as np
from PIL import Image, ImageDraw
import cv2
from cellpose import models
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
    QHBoxLayout, QCheckBox, QLabel, QMenu, QGraphicsTextItem, 
    QAction, QDialog
)
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QBrush, QPainter, QFont, QColor, QPalette, QImage
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QVariant
from superqt import QRangeSlider

class ImageViewer(QGraphicsView):
    def __init__(self, scene, roi_table, params):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.Antialiasing)
        self.roi_table = roi_table
        self.params = params

        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.live_path_item = None

        self._zoom = 0
        self._empty = True
        self._scene = scene

        self.drawing = False
        self.current_path = None
        self.temp_path_item = None
        self.current_points = []
        self.show_rois = True
        self.show_labels = True  # toggled by N
        self.roi_items = []      # list of QGraphicsPathItem
        self.roi_label_items = []# parallel list of QGraphicsTextItem
        self.path_pen = QPen(Qt.red, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.roi_opacity = 0.4

        self.brush_radius = 3
        self.cursor_brush = QGraphicsEllipseItem(0, 0, self.brush_radius * 2, self.brush_radius * 2)
        self.cursor_brush.setBrush(QBrush(Qt.blue))
        self.cursor_brush.setPen(QPen(Qt.NoPen))
        self.cursor_brush.setZValue(1000)  # Always on top
        self.cursor_brush.setVisible(False)
        self.scene().addItem(self.cursor_brush)

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

        low_item = QTableWidgetItem(str(self.params.low))
        high_item = QTableWidgetItem(str(self.params.high))
        self.roi_table.setItem(row, 2, low_item)
        self.roi_table.setItem(row, 3, high_item)

        self.roi_table.setItem(row, 4, QTableWidgetItem(str(0.5)))

    def get_random_color(self):
        r = random.randint(50, 255)
        g = random.randint(50, 255)
        b = random.randint(50, 255)
        return QColor(r, g, b)

def set_dark_theme(app):
    app.setStyle("Fusion")
    dark_palette = app.palette()

    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)

class Params:  # for global parameter management
    def __init__(self):
        self.low = 80
        self.high = 200
         
class MainWindow(QMainWindow):
    def __init__(self, preload_image=None):
        super().__init__()
        self.setWindowTitle('New RPOC Editor')
        self.params = Params() 
        self.loaded_img = None  # will store a grayscale numpy array

        # full right side
        self.roi_table = QTableWidget(0, 5)
        self.roi_table.setHorizontalHeaderLabels(['ROI Name', 'Coordinates', 'Lower Threshold', 'Upper Threshold', 'Modulation Level'])

        # top on left
        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        # middle left
        self.help_label = QLabel('Press "M" to toggle mask visibility\n'
                                 'Press "N" to toggle label visibility\n'
                                 'Right-click row in table to delete ROI\n'
                                 'Press "P" to preview the final mask')
        self.threshold_slider = QRangeSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue((20,80))
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.save_button = QPushButton("Save Mask")
        self.save_button.clicked.connect(self.save_mask)

        middle_controls_layout = QHBoxLayout()
        middle_controls_layout.addWidget(self.help_label)
        middle_controls_layout.addWidget(self.threshold_slider)
        middle_controls_layout.addWidget(self.save_button)

        # bottom left
        self.image_scene = QGraphicsScene()
        self.image_item = self.image_scene.addPixmap(QPixmap())  # Placeholder
        self.image_view = ImageViewer(self.image_scene, self.roi_table, self.params)   

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        left_layout.addWidget(load_button) 
        left_layout.addLayout(middle_controls_layout)
        left_layout.addWidget(self.image_view)

        layout.addLayout(left_layout)
        layout.addWidget(self.roi_table)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.roi_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.roi_table.customContextMenuRequested.connect(self.show_table_context_menu)

        preview_action = QAction("Preview Mask", self)
        preview_action.setShortcut("P")
        preview_action.triggered.connect(self.preview_mask)
        self.addAction(preview_action)

        self.cellpose_button = QPushButton("Segment with Cellpose")
        self.cellpose_button.clicked.connect(self.run_cellpose_segmentation)
        left_layout.addWidget(self.cellpose_button)

        if preload_image is not None:
            self.set_preloaded_image(preload_image)

    def run_cellpose_segmentation(self):
        if self.loaded_img is None:
            return
    
        self.cellpose_button.setText("Segmenting...")
        self.cellpose_button.setEnabled(False)
        QApplication.processEvents()

        low, high = self.threshold_slider.value()
        masked = self.loaded_img.copy()
        masked[(masked < low) | (masked > high)] = 0

        model = models.Cellpose(model_type='cyto3')
        masks, _, _, _ = model.eval([masked], diameter=None, channels=[0, 0])
        masks = masks[0]

        n_rois = len(self.image_view.roi_items)

        for mask_val in range(1, masks.max() + 1):
            binary_mask = np.uint8(masks == mask_val) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            contour = contours[0]
            path = QPainterPath()
            points = []

            for i, pt in enumerate(contour):
                x, y = pt[0]
                qpt = QPointF(x, y)
                points.append(qpt)
                if i == 0:
                    path.moveTo(qpt)
                else:
                    path.lineTo(qpt)
            path.closeSubpath()

            color = self.image_view.get_random_color()
            roi_item = self.image_scene.addPath(path, QPen(color, 2), QBrush(color))
            roi_item.setOpacity(self.image_view.roi_opacity if self.image_view.show_rois else 0)
            self.image_view.roi_items.append(roi_item)

            label_item = self.image_view.create_roi_label(n_rois + 1, points)
            self.image_view.roi_label_items.append(label_item)

            self.image_view.add_roi_to_table(n_rois + 1, points)
            n_rois += 1

        self.cellpose_button.setText("Segment with Cellpose")
        self.cellpose_button.setEnabled(True)


    def set_preloaded_image(self, pil_image):
        pil_image = pil_image.convert("RGB")
        img_array = np.array(pil_image)

        self.original_rgb_img = img_array
        self.loaded_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  

        self.update_displayed_image()  # apply current thresholds for display

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.jpg *.bmp)')
        if file_path:
            img_array = cv2.imread(file_path)
            if img_array is None:
                return
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            self.original_rgb_img = img_array
            self.loaded_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            self.update_displayed_image()

    def update_displayed_image(self):
        if not hasattr(self, 'original_rgb_img') or self.original_rgb_img is None:
            return

        low, high = self.threshold_slider.value()
        gray = self.loaded_img
        rgb = self.original_rgb_img.copy()

        mask_out = (gray < low) | (gray > high)
        rgb[mask_out] = [0, 0, 0]

        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        self.image_item.setPixmap(pixmap)
        self.image_scene.setSceneRect(QRectF(pixmap.rect()))

    def show_table_context_menu(self, pos):
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

        # relabel the remaining rois so they match table order
        for i, (roi, label) in enumerate(zip(viewer.roi_items, viewer.roi_label_items)):
            idx = i + 1  
            label.setPlainText(str(idx))

            path = roi.path()
            poly = path.toSubpathPolygons()[0]
            cx = sum(p.x() for p in poly) / len(poly)
            cy = sum(p.y() for p in poly) / len(poly)
            text_rect = label.boundingRect()
            label.setPos(cx - text_rect.width() / 2, cy - text_rect.height() / 2)

        # rebuild table
        viewer.roi_table.setRowCount(0)
        for i, roi in enumerate(viewer.roi_items):
            path = roi.path()
            points = path.toSubpathPolygons()[0]
            viewer.add_roi_to_table(i + 1, points)

    def on_threshold_changed(self, values):
        self.update_displayed_image()   # update visible pixel
        self.params.low, self.params.high = values

    def generate_final_mask(self):
        if self.loaded_img is None:
            return None

        height, width = self.loaded_img.shape[:2]
        final_mask = np.zeros((height, width), dtype=np.uint8)
        row_count = self.roi_table.rowCount()

        for row in range(row_count):
            idx_item = self.roi_table.item(row, 0)
            coords_item = self.roi_table.item(row, 1)
            low_item = self.roi_table.item(row, 2)
            high_item = self.roi_table.item(row, 3)
            mod_item = self.roi_table.item(row, 4)

            if not (idx_item and coords_item and low_item and high_item and mod_item):
                continue

            try:
                low_val = float(low_item.text())
                high_val = float(high_item.text())
                mod_val = float(mod_item.text())  # typically between 0.0 and 1.0
            except:
                continue

            roi_index = idx_item.data(Qt.UserRole)
            if roi_index is None:
                continue

            matching_item_idx = None
            for i, lbl in enumerate(self.image_view.roi_label_items):
                if lbl.toPlainText() == str(roi_index):
                    matching_item_idx = i
                    break
            if matching_item_idx is None:
                continue

            roi_path_item = self.image_view.roi_items[matching_item_idx]
            polygons = roi_path_item.path().toSubpathPolygons()
            if not polygons:
                continue

            polygon = polygons[0]
            pts = []
            for p in polygon:
                pts.append([int(round(p.x())), int(round(p.y()))])
            pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts], 255)

            inside_y, inside_x = np.where(roi_mask == 255)
            intensities = self.loaded_img[inside_y, inside_x]
            valid = (intensities >= low_val) & (intensities <= high_val)
            valid_idxs = np.where(valid)

            final_mask[inside_y[valid_idxs], inside_x[valid_idxs]] = int(mod_val * 255)

        return final_mask

    def preview_mask(self):
        mask = self.generate_final_mask()
        if mask is None:
            return

        height, width = mask.shape
        bytes_per_line = width
        qimg = QImage(mask.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg)

        dialog = QDialog(self)
        dialog.setWindowTitle("Mask Preview")
        layout = QVBoxLayout()
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.resize(width, height)
        dialog.exec_()

    def save_mask(self):
        if self.loaded_img is None:
            return 

        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Mask', '', 'PNG (*.png);;TIFF (*.tif);;All Files (*)')
        if not save_path:
            return

        final_mask = self.generate_final_mask()
        if final_mask is None:
            return

        cv2.imwrite(save_path, final_mask)
        print("Mask saved to:", save_path)

def launch_pyqt_editor(preloaded_image=None):
    app = QApplication.instance()
    app_created = False
    if app is None:
        app = QApplication(sys.argv)
        app_created = True

    set_dark_theme(app)
    win = MainWindow(preload_image=preloaded_image)
    win.resize(1200, 800)
    win.show()

    if app_created:
        app.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    set_dark_theme(app)  # dark mode
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
