import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_mask(image: np.ndarray) -> np.ndarray:
    data = image[0]
    h, w = data.shape

    seed = int(np.sum(data) % 1e6)
    rng = np.random.RandomState(seed)

    letter = chr(rng.randint(ord('A'), ord('Z') + 1))

    img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(img)

    font_size = int(min(w, h) * 0.7)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((w - text_w) // 2, (h - text_h) // 2)

    draw.text(pos, letter, fill=255, font=font)

    mask = np.array(img)
    binary_mask = (mask > 127).astype(np.uint8) * 255
    return binary_mask
