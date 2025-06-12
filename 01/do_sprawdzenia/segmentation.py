import numpy as np
from PIL import Image
import heapq

class WatershedVincentSoille:
    INIT = -2
    MASK = -3
    WSHED = 0
    INQUEUE = -4

    def __init__(self, image):
        self.image = image.convert("L")
        self.gray = np.array(self.image)
        self.height, self.width = self.gray.shape

    def transform(self):
        f = self.gray
        labels = np.full_like(f, self.INIT, dtype=np.int32)
        dist = np.zeros_like(f, dtype=np.int32)
        current_label = 0
        fifo = []

        sorted_pixels = [[] for _ in range(256)]
        for y in range(self.height):
            for x in range(self.width):
                sorted_pixels[f[y, x]].append((y, x))

        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)]

        for level in range(256):
            for y, x in sorted_pixels[level]:
                labels[y, x] = self.MASK
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if labels[ny, nx] >= 0:
                            dist[y, x] = 1
                            heapq.heappush(fifo, (1, y, x))
                            labels[y, x] = self.INQUEUE
                            break

            cur_dist = 1
            while fifo:
                d, y, x = heapq.heappop(fifo)
                if d > cur_dist:
                    cur_dist = d
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if labels[ny, nx] > 0:
                            if labels[y, x] == self.INQUEUE or labels[y, x] == self.WSHED:
                                labels[y, x] = labels[ny, nx]
                            elif labels[y, x] != labels[ny, nx]:
                                labels[y, x] = self.WSHED
                        elif labels[ny, nx] == self.MASK:
                            dist[ny, nx] = d + 1
                            heapq.heappush(fifo, (d + 1, ny, nx))
                            labels[ny, nx] = self.INQUEUE

            for y, x in sorted_pixels[level]:
                if labels[y, x] == self.MASK:
                    current_label += 1
                    labels[y, x] = current_label
                    queue = [(y, x)]
                    while queue:
                        qy, qx = queue.pop()
                        for dy, dx in neighbors:
                            ny, nx = qy + dy, qx + dx
                            if 0 <= ny < self.height and 0 <= nx < self.width:
                                if labels[ny, nx] == self.MASK:
                                    labels[ny, nx] = current_label
                                    queue.append((ny, nx))

        return self._visualize(labels)

    def _visualize(self, labels):
        from random import randint
        label_to_color = {
            self.WSHED: (0, 0, 0)
        }
        max_label = labels.max()
        for label in range(1, max_label + 1):
            label_to_color[label] = (randint(50, 255), randint(50, 255), randint(50, 255))

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            for x in range(self.width):
                label = labels[y, x]
                rgb[y, x] = label_to_color.get(label, (0, 0, 0))

        return Image.fromarray(rgb)