from PIL import Image

class BinManual:
    def __init__(self, image, threshold=150):
        self.image = image.convert("L")
        self.threshold = threshold

    def transform(self):
        return self.image.point(lambda p: 255 if p >= self.threshold else 0).convert("RGB")