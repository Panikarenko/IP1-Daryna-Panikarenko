import numpy as np
from PIL import Image
import math

class HoughTransform:
    def __init__(self, image, theta_density=1.0, skip_edge_detection=False):
        self.image = image.convert("L")
        self.theta_density = theta_density
        self.skip_edge_detection = skip_edge_detection

    def transform(self):
        width, height = self.image.size
        img_array = np.array(self.image)

        if not self.skip_edge_detection:
            from edge_filters import EdgeLaplacian
            edge = EdgeLaplacian(self.image, size=3)
            edge_img = edge.transform()
            img_array = np.array(edge_img.convert("L"))

        # ρmax = długość przekątnej obrazu
        rho_max = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))
        rho_range = 2 * rho_max + 1

        theta_size = int(180 * self.theta_density)
        thetas = np.linspace(0, np.pi, theta_size)
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        accumulator = np.zeros((theta_size, rho_range), dtype=np.uint64)

        # Znajdź współrzędne pikseli krawędzi
        y_idxs, x_idxs = np.nonzero(img_array > 0)  # tylko krawędzie!

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            rhos = (x * cos_t + y * sin_t).astype(int) + rho_max
            valid = (rhos >= 0) & (rhos < rho_range)
            accumulator[np.arange(theta_size)[valid], rhos[valid]] += 1

        acc_normalized = np.clip(accumulator * 255.0 / accumulator.max(), 0, 255).astype(np.uint8)
        hough_image = Image.fromarray(acc_normalized.T).convert("L")
        return hough_image
