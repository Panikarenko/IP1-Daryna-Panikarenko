import numpy as np
from PIL import Image

class BinOtsu:
    def __init__(self, image):
        self.image = image.convert("L")  # grayscale

    def transform(self):
        img_array = np.array(self.image)
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        total = img_array.size

        sumB = 0
        wB = 0
        maximum = 0.0
        sum1 = np.dot(np.arange(256), hist)
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum1 - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between > maximum:
                maximum = between
                threshold = t

        binarized = self.image.point(lambda p: 255 if p >= threshold else 0)
        return binarized.convert("RGB")