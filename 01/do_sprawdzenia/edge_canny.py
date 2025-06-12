from PIL import Image
import numpy as np
from scipy.ndimage import convolve, sobel
import math

class EdgeCanny:
    def __init__(self, image, lower_thresh=50, upper_thresh=100):
        self.image = image.convert("L")
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh

    def transform(self):
        img_array = np.asarray(self.image, dtype=np.float32)

        # 1. Rozmycie Gaussa 3x3, σ=1.6
        kernel = self.gaussian_kernel_3x3(sigma=1.6)
        blurred = convolve(img_array, kernel)

        # 2. Gradienty Sobela
        gx = sobel(blurred, axis=1)
        gy = sobel(blurred, axis=0)

        # 3. Moc i kierunek gradientu
        magnitude = np.hypot(gx, gy)
        direction = np.arctan2(gy, gx)

        # 4. Tłumienie niemaksymalne
        nms = self.non_maximum_suppression(magnitude, direction)

        # 5. Progowanie z histerezą
        edges = self.hysteresis(nms, direction)

        return Image.fromarray((edges * 255).astype(np.uint8))

    def gaussian_kernel_3x3(self, sigma):
        ax = np.linspace(-1, 1, 3)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def non_maximum_suppression(self, magnitude, direction):
        height, width = magnitude.shape
        output = np.zeros((height, width), dtype=np.float32)
        angle = direction * 180. / np.pi
        angle[angle < 0] += 180

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                q = 255
                r = 255
                a = angle[y, x]

                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = magnitude[y, x + 1]
                    r = magnitude[y, x - 1]
                elif (22.5 <= a < 67.5):
                    q = magnitude[y + 1, x - 1]
                    r = magnitude[y - 1, x + 1]
                elif (67.5 <= a < 112.5):
                    q = magnitude[y + 1, x]
                    r = magnitude[y - 1, x]
                elif (112.5 <= a < 157.5):
                    q = magnitude[y - 1, x - 1]
                    r = magnitude[y + 1, x + 1]

                # Dodatkowy warunek: moc gradientu >= upper_thresh
                if magnitude[y, x] >= q and magnitude[y, x] >= r and magnitude[y, x] >= self.upper_thresh:
                    output[y, x] = magnitude[y, x]

        return output

    def hysteresis(self, nms, direction):
        height, width = nms.shape
        visited = np.zeros((height, width), dtype=bool)
        output = np.zeros((height, width), dtype=bool)

        def is_valid(x, y):
            return 0 <= x < width and 0 <= y < height

        def get_direction_sector(angle):
            angle_deg = angle * 180. / np.pi
            if angle_deg < 0:
                angle_deg += 180
            if (0 <= angle_deg < 22.5) or (157.5 <= angle_deg <= 180):
                return (1, 0)
            elif 22.5 <= angle_deg < 67.5:
                return (1, -1)
            elif 67.5 <= angle_deg < 112.5:
                return (0, -1)
            elif 112.5 <= angle_deg < 157.5:
                return (-1, -1)
            else:
                return (1, 0)

        def follow_edge(x, y):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if not is_valid(cx, cy) or visited[cy, cx]:
                    continue

                visited[cy, cx] = True
                output[cy, cx] = True

                current_dir = get_direction_sector(direction[cy, cx])
                dx, dy = current_dir

                for d in [-1, 1]:
                    nx, ny = cx + dx * d, cy + dy * d
                    if not is_valid(nx, ny) or visited[ny, nx]:
                        continue

                    if nms[ny, nx] < self.lower_thresh:
                        continue

                    neighbor_dir = get_direction_sector(direction[ny, nx])
                    if neighbor_dir != current_dir:
                        continue

                    sx, sy = nx + dx, ny + dy
                    sx2, sy2 = nx - dx, ny - dy
                    strong_enough = True
                    if is_valid(sx, sy) and nms[ny, nx] < nms[sy, sx]:
                        strong_enough = False
                    if is_valid(sx2, sy2) and nms[ny, nx] < nms[sy2, sx2] and (sx2, sy2) != (cx, cy):
                        strong_enough = False
                    if not strong_enough:
                        continue

                    stack.append((nx, ny))

        for y in range(height):
            for x in range(width):
                if nms[y, x] >= self.upper_thresh and not visited[y, x]:
                    follow_edge(x, y)

        return output