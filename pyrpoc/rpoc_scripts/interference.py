import numpy as np
from PIL import Image

def generate_mask(image: np.ndarray) -> np.ndarray:
    data = image[0]
    h, w = data.shape

    seed = int(np.sum(data) % 1e6)
    rng = np.random.RandomState(seed)

    num_waves = 3
    angles = np.linspace(0, 2 * np.pi, num_waves, endpoint=False) + rng.uniform(0, 2*np.pi)
    frequencies = rng.uniform(0.01, 0.03, size=num_waves)  # spatial frequency per wave

    yy, xx = np.indices((h, w))
    xx = xx - w // 2
    yy = yy - h // 2

    pattern = np.zeros((h, w))
    for theta, freq in zip(angles, frequencies):
        kx = np.cos(theta) * freq
        ky = np.sin(theta) * freq
        pattern += np.cos(kx * xx + ky * yy)

    pattern -= pattern.min()
    pattern /= pattern.max()

    pattern = np.power(pattern, 1.5)  
    binary_mask = (pattern > 0.5).astype(np.uint8) * 255

    return binary_mask
