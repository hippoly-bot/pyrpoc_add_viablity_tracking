import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def stitch(base_dir, channel=0, threshold=230, output_filename="stitched_mosaic.tif"):
    with open(os.path.join(base_dir, "mosaic_metadata.json"), "r") as f:
        metadata = json.load(f)

    rows = metadata["rows"]
    cols = metadata["cols"]
    overlap = metadata["overlap"]
    tile_order = metadata["tile_order"]

    tile_dir = os.path.join(base_dir, "tiles")
    sample_tile_path = os.path.join(tile_dir, f"tile_0_0_ch{channel}.tif")
    sample_tile = np.array(Image.open(sample_tile_path))
    tile_h, tile_w = sample_tile.shape

    step_h = int(tile_h * (1 - overlap))
    step_w = int(tile_w * (1 - overlap))
    mosaic_h = step_h * (rows - 1) + tile_h
    mosaic_w = step_w * (cols - 1) + tile_w

    mosaic = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    weight = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

    for i, j in tile_order:
        tile_path = os.path.join(tile_dir, f"tile_{i}_{j}_ch{channel}.tif")

        tile = np.array(Image.open(tile_path)).astype(np.float32)

        # tile[tile > threshold] = 0.0

        y1, x1 = i * step_h, j * step_w
        y2, x2 = y1 + tile_h, x1 + tile_w

        valid = tile > 0
        mosaic[y1:y2, x1:x2] += tile * valid
        weight[y1:y2, x1:x2] += valid.astype(np.float32)

    nonzero = weight > 0
    stitched = np.zeros_like(mosaic)
    stitched[nonzero] = mosaic[nonzero] / weight[nonzero]

    stitched_img = (np.clip(stitched, 0, 255)).astype(np.uint8)
    plt.imshow(stitched_img)
    plt.show()

    Image.fromarray(stitched_img).save(os.path.join(base_dir, output_filename))

stitch(r"C:\Users\Lab Admin\Box\(L2 Sensitive) zhan2017\Zhang lab data\Ishaan\seohee 1", channel=0)
