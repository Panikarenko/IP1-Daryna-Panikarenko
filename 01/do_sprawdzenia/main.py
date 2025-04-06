import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QFileDialog, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtCore import Qt
from PIL import Image, ImageQt
from PyQt6.QtWidgets import QSpacerItem, QSizePolicy
from scipy.interpolate import CubicSpline
import numpy as np

class Correction:
    def __init__(self, image, shift=0, factor=1.0, gamma=1.0):
        self.image = image.convert("RGB")
        self.shift = shift
        self.factor = factor
        self.gamma = gamma
        self.LUT = list(range(256))  # startowa LUT

    def transform(self):
        # LUT base points: from 0 to 255
        x = np.arange(256)

        # Start with identity LUT
        y = np.array(x, dtype=np.float32)

        # Krzywimy LUT w centralnym zakresie (np. 5 punktów wokół 128)
        control_x = [0, 64, 128, 192, 255]
        control_y = [0, 
             64 + self.shift * 0.4, 
             128 + self.shift * 0.8, 
             192 + self.shift * 0.4, 
             255]

        # Make sure values are clamped in [0, 255]
        control_y = np.clip(control_y, 0, 255)

        # Interpolacja spline
        cs = CubicSpline(control_x, control_y)
        lut = np.clip(cs(x), 0, 255).astype(np.uint8)

        # Gamma i kontrast
        adjusted_lut = []
        for val in lut:
            val = val * self.factor
            val = max(0.0, val / 255.0)
            val = pow(val, self.gamma) * 255
            val = max(0, min(255, int(val)))
            adjusted_lut.append(val)

        # Zastosowanie LUT
        pixels = self.image.load()
        width, height = self.image.size
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                pixels[x, y] = (
                    adjusted_lut[r],
                    adjusted_lut[g],
                    adjusted_lut[b]
                )

        return self.image

class ConversionGrayscale:
    PIXEL_VAL_MAX = 255
    PIXEL_VAL_MIN = 0

    def __init__(self, image):
        self.image = image.convert("RGB")

    def transform(self):
        pixels = self.image.load()
        width, height = self.image.size

        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]

                # Jeśli obraz czarno-biały (piksele równe 0 lub 255 we wszystkich kanałach)
                if (r == g == b) and (r == self.PIXEL_VAL_MAX or r == self.PIXEL_VAL_MIN):
                    gray = self.PIXEL_VAL_MAX if r == self.PIXEL_VAL_MAX else self.PIXEL_VAL_MIN
                else:
                    # Konwersja do szarości z wagami (bardziej realistyczna percepcja)
                    gray = int(0.3 * r + 0.6 * g + 0.1 * b)
                    gray = max(self.PIXEL_VAL_MIN, min(self.PIXEL_VAL_MAX, gray))  # zabezpieczenie

                pixels[x, y] = (gray, gray, gray)

        return self.image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PHOTOSHOP")
        self.setGeometry(100, 100, 800, 600)

        # Create a menu bar
        menu_bar = self.menuBar()

        # Create a file menu
        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        # Create a central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Create a side panel widget and layout
        self.side_panel = QWidget()
        self.side_panel_layout = QVBoxLayout()
        self.side_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.side_panel.setLayout(self.side_panel_layout)
        self.layout.addWidget(self.side_panel)
        self.side_panel.setFixedWidth(200)

        cwiczenia1_menu = QMenu("Ćwiczenia 1", self)
        odcienie_szarosci = QAction("Odcienie szarości", self)
        cwiczenia1_menu.addAction(odcienie_szarosci)
        lut = QAction("Jasność + kontrast + gamma", self)
        cwiczenia1_menu.addAction(lut)
        odcienie_szarosci.triggered.connect(self.odcienie_szarosci_triggered)
        lut.triggered.connect(self.lut_triggered)

        # Add the side menu to the side panel
        button1 = QPushButton("Ćwiczenie 1")
        button1.setMenu(cwiczenia1_menu)
        self.side_panel_layout.addWidget(button1)
        self.action2_button = QPushButton("Action 2")
        self.side_panel_layout.addWidget(self.action2_button)

        # Create a QLabel to display the image
        self.image_label = QLabel(self)
        self.image_label.setFixedWidth(600)
        self.image_label.setFixedHeight(600)
        self.layout.addWidget(self.image_label)

        # Add actions to the file menu
        open_action = QAction("Open", self)
        save_action = QAction("Save", self)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        # Open action
        open_action.triggered.connect(self.open_action_triggered)
        save_action.triggered.connect(self.save_action_triggered)

        # Widget z suwakami LUT
        self.lut_sliders_widget = QWidget()
        self.lut_sliders_layout = QVBoxLayout()
        self.lut_sliders_widget.setLayout(self.lut_sliders_layout)

        self.brightness_slider = self.create_slider("Jasność")
        self.contrast_slider = self.create_slider("Kontrast")
        self.gamma_slider = self.create_slider("Gamma")

        self.lut_sliders_layout.addWidget(self.brightness_slider['label'])
        self.lut_sliders_layout.addWidget(self.brightness_slider['slider'])
        self.lut_sliders_layout.addWidget(self.contrast_slider['label'])
        self.lut_sliders_layout.addWidget(self.contrast_slider['slider'])
        self.lut_sliders_layout.addWidget(self.gamma_slider['label'])
        self.lut_sliders_layout.addWidget(self.gamma_slider['slider'])

        #RESETUJ START
        # Spacer wypychający zawartość w górę
        spacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self.side_panel_layout.addItem(spacer)

        # Przycisk Resetuj obraz
        reset_button = QPushButton("Resetuj obraz")
        reset_button.clicked.connect(self.reset_image)
        reset_button.setStyleSheet("""
            QPushButton {
                background-color: #37a7ec;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #7a9cb0;
            }
        """)
        self.side_panel_layout.addWidget(reset_button)
        #RESETUJ END

        self.lut_sliders_widget.setVisible(False)

        # Dodajemy suwaki przed "Action 2"
        index = self.side_panel_layout.indexOf(self.action2_button)
        self.side_panel_layout.insertWidget(index, self.lut_sliders_widget)
        
    def apply_lut_correction(self):
        if hasattr(self, 'current_image'):
            shift = self.brightness_slider["slider"].value()
            contrast_val = self.contrast_slider["slider"].value()
            gamma_val = self.gamma_slider["slider"].value()

            factor = 1.0 + (contrast_val / 100.0)
            gamma = 1.0 + (gamma_val / 100.0)

            corrected = Correction(self.current_image.copy(), shift, factor, gamma)
            self.image = corrected.transform()
            self.display_image()

    def create_slider(self, name):
        label = QLabel(f"{name}: 0")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.valueChanged.connect(lambda value, l=label, n=name: l.setText(f"{n}: {value}"))
        return {"label": label, "slider": slider}

    def open_action_triggered(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "PPM Files (*.ppm);;All Files (*)")
        if file_name:
            try:
                self.image = Image.open(file_name).convert("RGB")
                self.original_image = self.image.copy()
                self.current_image = self.image.copy()
                self.display_image()
            except Exception as e:
                print(f"Failed to open image: {e}")

    def save_action_triggered(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "PPM Files (*.ppm);;All Files (*)")
        if file_name:
            if not file_name.lower().endswith(".ppm"):
                file_name += ".ppm"
            try:
                self.image.save(file_name)
            except Exception as e:
                print(f"Failed to save image: {e}")

    def edit_pixels(self):
        # Example: Invert the colors of the image
        pixels = self.image.load()
        for i in range(self.image.width):
            for j in range(self.image.height):
                r, g, b = pixels[i, j]
                pixels[i, j] = (255 - r, 255 - g, 255 - b)

    def display_image(self):
        # Convert the PIL image to QPixmap and display it
        self.image = self.image.convert("RGB")
        qt_image = ImageQt.ImageQt(self.image)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def odcienie_szarosci_triggered(self):
        if hasattr(self, 'image'):
            converter = ConversionGrayscale(self.image)
            self.image = converter.transform()
            self.current_image = self.image.copy()  # ← zapisz nowy stan
            self.display_image()

    def lut_triggered(self):
        self.lut_sliders_widget.setVisible(not self.lut_sliders_widget.isVisible())
        # Podpinamy update dla suwaków
        for slider_info in [self.brightness_slider, self.contrast_slider, self.gamma_slider]:
            slider_info["slider"].valueChanged.connect(self.apply_lut_correction)

    def apply_lut_correction(self):
        if hasattr(self, 'original_image'):
            shift = self.brightness_slider["slider"].value()
            contrast_val = self.contrast_slider["slider"].value()
            gamma_val = self.gamma_slider["slider"].value()

            factor = 1.0 + (contrast_val / 100.0)
            gamma = 1.0 + (gamma_val / 100.0)

            # użyj oryginału!
            corrected = Correction(self.current_image.copy(), shift, factor, gamma)
            self.image = corrected.transform()
            self.display_image()

    def reset_image(self):
        if hasattr(self, 'original_image'):
            self.image = self.original_image.copy()
            self.current_image = self.original_image.copy()
            self.display_image()
            self.brightness_slider["slider"].setValue(0)
            self.contrast_slider["slider"].setValue(0)
            self.gamma_slider["slider"].setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())