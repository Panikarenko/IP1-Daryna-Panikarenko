import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

class BinNiblack:
    def __init__(self, image, radius=3, k=-0.2):
        self.image = image.convert("L")
        self.radius = radius
        self.k = k

    def transform(self):
        img_array = np.array(self.image, dtype=np.float32)

        mean = uniform_filter(img_array, size=2 * self.radius + 1)
        mean_sq = uniform_filter(img_array**2, size=2 * self.radius + 1)

        stddev = np.sqrt(mean_sq - mean**2)

        threshold = mean + self.k * stddev

        binary = (img_array >= threshold).astype(np.uint8) * 255
        return Image.fromarray(binary).convert("RGB")