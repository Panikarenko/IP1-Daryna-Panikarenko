import numpy as np
from PIL import Image
import math
from scipy.signal import convolve2d
from scipy.ndimage import convolve

class BlurGaussian:
    @staticmethod
    def getGauss(x, y, sigma):
        coefficient = 1 / (2 * math.pi * sigma ** 2)
        exponent = -(x ** 2 + y ** 2) / (2 * sigma ** 2)
        return coefficient * math.exp(exponent)

class EdgeGradient:
    def __init__(self, image):
        self.image = image.convert("RGB")
        self.pixels = np.array(self.image, dtype=np.float32)
        self.g_x = None
        self.g_y = None

    def horizontalDetection(self, channel):
        return convolve2d(channel, self.g_x, mode="same", boundary="symm")

    def verticalDetection(self, channel):
        return convolve2d(channel, self.g_y, mode="same", boundary="symm")

    def transform(self):
        # Rozdziel na kanały
        r, g, b = self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2]

        new_channels = []
        for channel in [r, g, b]:
            gx = self.horizontalDetection(channel)
            gy = self.verticalDetection(channel)
            mag = np.sqrt(gx**2 + gy**2)
            mag = np.clip(mag, 0, 255)
            new_channels.append(mag.astype(np.uint8))

        # Połącz z powrotem w RGB
        result_rgb = np.stack(new_channels, axis=2)
        return Image.fromarray(result_rgb, mode="RGB")
    
class EdgeRoberts(EdgeGradient):
    def __init__(self, image):
        super().__init__(image)
        self.g_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        self.g_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

class EdgePrewitt(EdgeGradient):
    def __init__(self, image):
        super().__init__(image)
        self.g_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
        self.g_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], dtype=np.float32)

class EdgeSobel(EdgeGradient):
    def __init__(self, image):
        super().__init__(image)
        self.g_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)
        self.g_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]], dtype=np.float32)

class EdgeLaplacian:
    def __init__(self, image, size=3):
        self.image = image.convert("RGB")
        self.pixels = np.array(self.image, dtype=np.float32)
        self.mask = self.getMask(size)

    def getMask(self, size):
        assert size % 2 == 1, "Size must be odd"
        mask = -1 * np.ones((size, size), dtype=np.float32)
        center = size // 2
        mask[center, center] = size * size - 1
        return mask

    def transform(self):
        r, g, b = self.pixels[:, :, 0], self.pixels[:, :, 1], self.pixels[:, :, 2]
        new_channels = []
        for channel in [r, g, b]:
            response = convolve2d(channel, self.mask, mode="same", boundary="symm")
            response = np.clip(np.abs(response), 0, 255)
            new_channels.append(response.astype(np.uint8))
        result_rgb = np.stack(new_channels, axis=2)
        return Image.fromarray(result_rgb, mode="RGB")
    
class EdgeLaplaceOfGauss:
    def __init__(self, image, size=3, sigma=1.6, threshold=5):
        self.image = image.convert("L")  # tylko jasności
        self.pixels = np.array(self.image, dtype=np.float32)
        self.size = size
        self.sigma = sigma
        self.threshold = threshold
        self.mask = self.getMask(size, sigma)

    def getGauss(self, x, y, sigma):
        return np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    def getLoG(self, x, y, sigma):
        factor = (x**2 + y**2 - 2 * sigma**2) / (sigma**4)
        return factor * self.getGauss(x, y, sigma)

    def getMask(self, size, sigma):
        assert size % 2 == 1
        half = size // 2
        mask = np.zeros((size, size), dtype=np.float32)
        for i in range(size):
            for j in range(size):
                x = i - half
                y = j - half
                mask[i, j] = self.getLoG(x, y, sigma)
        return mask

    def transform(self):
        log_image = convolve2d(self.pixels, self.mask, mode="same", boundary="symm")
        
        # Normalizacja do zakresu [0, 255], bo wartości po splotach mogą być np. [-50, 70]
        log_image = log_image - log_image.min()
        log_image = (log_image / log_image.max()) * 255

        output = np.zeros_like(log_image, dtype=np.uint8)
        v0 = 128  # Zgodnie z instrukcją – środek 0–255
        h, w = log_image.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                window = log_image[i-1:i+2, j-1:j+2]
                min_val = window.min()
                max_val = window.max()
                if min_val < v0 - self.threshold and max_val > v0 + self.threshold:
                    output[i, j] = 255

        return Image.fromarray(output, mode="L")