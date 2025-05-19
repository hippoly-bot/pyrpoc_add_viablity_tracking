import numpy as np

def generate_mask(data_list):
    image = data_list[0]
    mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            mask[i,j] = 255
    return mask