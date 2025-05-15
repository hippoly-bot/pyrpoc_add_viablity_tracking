#!/usr/bin/env python3
"""
Standalone pathology viewer for large TIFF mosaics.
Loads individual tile TIFFs and metadata JSON to render a zoomable, pannable view,
with seamless blending in overlapping regions.
"""
import sys
import os
import json
import glob
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class PathologyViewer(QGraphicsView):
    def __init__(self, metadata_path):
        super().__init__()
        self.setWindowTitle("Pathology Mosaic Viewer")
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.load_metadata(metadata_path)
        self.build_canvas()
        self.load_tiles_with_blending()
        self.display_canvas()
        self.showMaximized()

    def load_metadata(self, metadata_path):
        with open(metadata_path, 'r') as f:
            md = json.load(f)
        self.rows = md.get('rows')
        self.cols = md.get('cols')
        self.overlap = md.get('overlap', 0.0)
        self.tile_order = md.get('tile_order', [])
        self.meta_dir = os.path.dirname(os.path.abspath(metadata_path))
        self.tile_dir = os.path.join(self.meta_dir, 'tiles')
        first = self.tile_order[0]
        sample_path = os.path.join(self.tile_dir, f"tile_{first[0]}_{first[1]}_ch0.tif")
        im = Image.open(sample_path)
        self.tile_w, self.tile_h = im.size
        self.step_px = int(self.tile_w * (1 - self.overlap))

    def build_canvas(self):
        mosaic_w = self.step_px * (self.cols - 1) + self.tile_w
        mosaic_h = self.step_px * (self.rows - 1) + self.tile_h
        self.canvas_rgb = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.float32)
        self.weight_map = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

    def load_tiles_with_blending(self):
        for i, j in self.tile_order:
            x = j * self.step_px
            y = i * self.step_px
            path = os.path.join(self.tile_dir, f"tile_{i}_{j}_ch0.tif")
            if not os.path.exists(path):
                continue
            tile = np.array(Image.open(path)).astype(np.float32)
            tile_rgb = np.stack([tile]*3, axis=-1) / 255.0
            h, w = tile.shape

            # Create a weight mask with raised cosine ramp in overlap
            yy, xx = np.meshgrid(np.linspace(-1,1,h), np.linspace(-1,1,w), indexing='ij')
            ramp = 0.5 * (1 + np.cos(np.pi * np.clip(np.maximum(np.abs(xx), np.abs(yy)), 0, 1)))
            ramp = ramp.astype(np.float32)

            self.canvas_rgb[y:y+h, x:x+w, :] += tile_rgb * ramp[..., None]
            self.weight_map[y:y+h, x:x+w] += ramp

    def display_canvas(self):
        norm_weight = np.clip(self.weight_map, 1e-6, None)
        blended = self.canvas_rgb / norm_weight[..., None]
        rgb8 = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        h, w = rgb8.shape[:2]
        img = QImage(rgb8.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.scene.addPixmap(pix)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

if __name__ == '__main__':
    path = r"C:\Users\Lab Admin\Box\(L2 Sensitive) zhan2017\Zhang lab data\Ishaan\5-06-2025_mosaic2\mosaic_metadata.json"
    app = QApplication(sys.argv)
    viewer = PathologyViewer(path)
    viewer.show()
    sys.exit(app.exec_())
