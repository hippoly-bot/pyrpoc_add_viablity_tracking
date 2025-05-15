import numpy as np
import matplotlib.pyplot as plt
from letters import generate_mask  # from [script name] import generate_mask

def create_test_images():
    images = []

    uniform = np.ones((1, 512, 512), dtype=np.uint8) * 100
    images.append(("Uniform 100", uniform))

    gradient = np.linspace(0, 255, 512).astype(np.uint8)
    gradient_image = np.tile(gradient, (512, 1))[None, ...]
    images.append(("Horizontal Gradient", gradient_image))

    noise = np.random.randint(0, 256, (1, 512, 512), dtype=np.uint8)
    images.append(("Random Noise", noise))

    shape = np.zeros((512, 512), dtype=np.uint8)
    shape[128:384, 128:384] = 200
    images.append(("Centered Square", shape[None, ...]))

    sparse = np.zeros((512, 512), dtype=np.uint8)
    sparse[60:90, 60:70] = 255
    sparse[400:410, 400:410] = 255
    images.append(("Sparse Bright Pixels", sparse[None, ...]))

    return images

def test_generate_mask():
    test_cases = create_test_images()

    fig, axs = plt.subplots(len(test_cases), 2, figsize=(6, 3 * len(test_cases)))
    if len(test_cases) == 1:
        axs = [axs] 

    for idx, (title, img) in enumerate(test_cases):
        mask = generate_mask(img)

        axs[idx][0].imshow(img[0], cmap='gray')
        axs[idx][0].set_title(f"Input: {title}")
        axs[idx][0].axis('off')

        axs[idx][1].imshow(mask, cmap='gray')
        axs[idx][1].set_title("Generated Binary Mask")
        axs[idx][1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_generate_mask()
