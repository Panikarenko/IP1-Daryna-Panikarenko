# from PIL import Image
# import numpy as np
# from scipy.ndimage import gaussian_filter, sobel, maximum_filter

# class CornerHarris:
#     def __init__(self, image, sigma=1.6, sigma_weight=0.76, k_param=0.05, threshold=3e7):
#         self.image = image.convert("L")  # grayscale
#         self.sigma = sigma
#         self.sigma_weight = sigma_weight
#         self.k_param = k_param
#         self.threshold = threshold

#     def transform(self):
#         img = np.array(self.image, dtype=np.float32)
#         img = gaussian_filter(img, sigma=self.sigma)  # blur input

#         # Oblicz gradienty Sobela
#         Gx = sobel(img, axis=1)
#         Gy = sobel(img, axis=0)

#         # Iloczyny gradientów
#         Ixx = Gx * Gx
#         Iyy = Gy * Gy
#         Ixy = Gx * Gy

#         # Rozmycie iloczynów Gaussowsko
#         Sxx = gaussian_filter(Ixx, sigma=self.sigma) * self.sigma_weight
#         Syy = gaussian_filter(Iyy, sigma=self.sigma) * self.sigma_weight
#         Sxy = gaussian_filter(Ixy, sigma=self.sigma) * self.sigma_weight

#         # Oblicz miarę Harrisową
#         det = Sxx * Syy - Sxy * Sxy
#         trace = Sxx + Syy
#         R = det - self.k_param * (trace ** 2)

#         # Progowanie
#         corner_candidates = (R > self.threshold) * R

#         # NMS: non-maximum suppression
#         max_filtered = maximum_filter(corner_candidates, size=(3, 3))
#         corners = (corner_candidates == max_filtered) & (corner_candidates > 0)

#         # Wynik binarny: 255 dla narożników
#         result = np.zeros_like(img, dtype=np.uint8)
#         result[corners] = 255

#         # Zamiana na obraz RGB do dalszego przetwarzania
#         return Image.fromarray(result).convert("RGB")














from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, sobel
import math

def get_gauss(k, l, sigma):
    return (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(k ** 2 + l ** 2) / (2 * sigma ** 2))

class CornerHarris:
    def __init__(self, image, sigma=1.6, sigma_weight=0.76, k_param=0.05, threshold=3e7):
        self.image = image.convert("L")  # grayscale
        self.sigma = sigma
        self.sigma_weight = sigma_weight
        self.k_param = k_param
        self.threshold = threshold

    def transform(self):
        img = np.array(self.image, dtype=np.float32)
        height, width = img.shape

        img = gaussian_filter(img, sigma=self.sigma)  # blur input

        Gx = sobel(img, axis=1)
        Gy = sobel(img, axis=0)

        Ixx = Gx * Gx
        Iyy = Gy * Gy
        Ixy = Gx * Gy

        Sxx = np.zeros_like(Ixx)
        Syy = np.zeros_like(Iyy)
        Sxy = np.zeros_like(Ixy)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        weight = get_gauss(k, l, self.sigma)
                        Sxx[i, j] += Ixx[i + k, j + l] * weight
                        Syy[i, j] += Iyy[i + k, j + l] * weight
                        Sxy[i, j] += Ixy[i + k, j + l] * weight

        Sxx *= self.sigma_weight
        Syy *= self.sigma_weight
        Sxy *= self.sigma_weight

        R = np.zeros_like(img)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                det = Sxx[i, j] * Syy[i, j] - Sxy[i, j] ** 2
                trace = Sxx[i, j] + Syy[i, j]
                R[i, j] = det - self.k_param * (trace ** 2)

        corner_candidates = np.zeros_like(R)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if R[i, j] > self.threshold:
                    corner_candidates[i, j] = R[i, j]

        search = True
        while search:
            search = False
            corner_nms = np.zeros_like(corner_candidates)

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    val = corner_candidates[i, j]
                    if val == 0:
                        continue
                    window = corner_candidates[i - 1:i + 2, j - 1:j + 2]
                    if val == np.max(window):
                        corner_nms[i, j] = val
                    else:
                        search = True

            corner_candidates = corner_nms

        # Final binary result: 1 = corner, 0 = not a corner
        result = np.zeros_like(img, dtype=np.uint8)
        result[corner_candidates > 0] = 1

        return Image.fromarray(result * 255).convert("RGB")
    

