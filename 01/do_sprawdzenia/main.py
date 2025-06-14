import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMenu, QFileDialog, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider
from PyQt6.QtGui import QAction, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal
from PIL import Image, ImageQt
from PyQt6.QtWidgets import QSpacerItem, QSizePolicy
from scipy.interpolate import CubicSpline
import numpy as np
from PyQt6.QtWidgets import QCheckBox, QDialog, QVBoxLayout, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from edge_filters import EdgeRoberts, EdgePrewitt, EdgeSobel, EdgeLaplacian, EdgeLaplaceOfGauss
import cv2

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

        adjusted_lut = []
        for val in lut:
            # KONSTRAST
            normalized_value = self.factor - 1.0
            movable_values_ids = [1, 2, 4, 5]
            xp = [0, 50, 100, 128, 155, 205, 255]
            fp = [0, 50, 100, 128, 155, 205, 255]
            for val_id in movable_values_ids:
                change = min(fp[val_id], 255-fp[val_id]) * normalized_value
                if val_id == 1 or val_id == 5:
                    change *= 2
                new_val = fp[val_id] - change if xp[val_id] < 128 else fp[val_id] + change
                new_val = min(128, new_val) if xp[val_id] < 128 else max(128, new_val)
                if new_val < 0:
                    new_val = 0
                elif new_val > 255:
                    new_val = 255
                fp[val_id] = new_val

            val = np.interp(val, xp, fp)

            # GAMMA
            nx = 255-(self.gamma * 127.0)
            ny = 255-nx
            xp = [0, nx, 255]
            fp = [0, ny, 255]
            val = np.interp(val, xp, fp)

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

class HistogramDialog(QDialog):
    image_modified = pyqtSignal(object)

    def __init__(self, pil_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.setMinimumSize(500, 400)
        self.main_layout = QVBoxLayout(self)

        self.pil_image = pil_image.copy()
        self.arr = np.array(self.pil_image)

        self.hist_checkboxes = {}
        for channel in ['R', 'G', 'B', 'L']:
            checkbox = QCheckBox(channel)
            checkbox.setChecked(True)
            self.hist_checkboxes[channel] = checkbox
            self.main_layout.addWidget(checkbox)

        self.hist_stretch_button = QPushButton("Rozciąganie hist. obrazu")
        self.hist_equalize_button = QPushButton("Wyrównywanie hist. obrazu")
        self.main_layout.addWidget(self.hist_stretch_button)
        self.main_layout.addWidget(self.hist_equalize_button)

        self.hist_stretch_button.clicked.connect(self.stretch_histogram)
        self.hist_equalize_button.clicked.connect(self.equalize_histogram)
        
        apply_btn = QPushButton("Zastosuj i zamknij")
        apply_btn.clicked.connect(self.apply_and_close)
        self.main_layout.addWidget(apply_btn)

        for checkbox in self.hist_checkboxes.values():
            checkbox.stateChanged.connect(lambda _, l=self.main_layout: self.plot_histogram(l))

        self.plot_histogram(self.main_layout)

    def plot_histogram(self, layout):
        if hasattr(self, 'canvas'):
            layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            del self.canvas

        r = self.arr[:, :, 0].flatten()
        g = self.arr[:, :, 1].flatten()
        b = self.arr[:, :, 2].flatten()
        l = (0.3 * r + 0.6 * g + 0.1 * b).astype(np.uint8)

        fig, ax = plt.subplots()
        if self.hist_checkboxes['R'].isChecked():
            ax.hist(r, bins=256, color='red', alpha=0.5, label='R', density=False)
        else:
            ax.hist(r, bins=256, color='white', alpha=0.0, label='R', density=False)
        if self.hist_checkboxes['G'].isChecked():
            ax.hist(g, bins=256, color='green', alpha=0.5, label='G', density=False)
        else:
            ax.hist(g, bins=256, color='white', alpha=0.0, label='G', density=False)
        if self.hist_checkboxes['B'].isChecked():
            ax.hist(b, bins=256, color='blue', alpha=0.5, label='B', density=False)
        else:
            ax.hist(b, bins=256, color='white', alpha=0.0, label='B', density=False)
        if self.hist_checkboxes['L'].isChecked():
            ax.hist(l, bins=256, color='gray', alpha=0.5, label='L', density=False)
        else:
            ax.hist(l, bins=256, color='white', alpha=0.0, label='L', density=False)
        ax.set_xlim([0, 255])
        ax.set_ylim(top=None)
        ax.set_title("Histogram RGB/L")
        ax.legend()

        self.canvas = FigureCanvas(fig)
        layout.insertWidget(layout.count() - 1, self.canvas)

    def stretch_histogram(self):
        arr = self.arr.astype(np.float32)
        arr_stretched = arr.copy()

        for c in range(3):
            channel = arr[:, :, c]
            min_val = channel.min()
            max_val = channel.max()

            if max_val > min_val:
                stretched = (channel - min_val) * 255.0 / (max_val - min_val)
                arr_stretched[:, :, c] = np.clip(stretched, 0, 255)

        self.arr = arr_stretched.astype(np.uint8)
        self.plot_histogram(self.main_layout)
        self.image_modified.emit(Image.fromarray(self.arr))

    def equalize_histogram(self):
        arr = self.arr
        arr_eq = arr.copy()
        for c in range(3):
            channel = arr[:, :, c]
            hist, bins = np.histogram(channel.flatten(), 256, [0,256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            arr_eq[:, :, c] = np.interp(channel.flatten(), bins[:-1], cdf_normalized).reshape(channel.shape).astype(np.uint8)
        self.arr = arr_eq
        self.plot_histogram(self.main_layout)
        self.image_modified.emit(Image.fromarray(self.arr))

    def apply_and_close(self):
        self.image_modified.emit(Image.fromarray(self.arr))
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        self.ignoring_binary_update = False
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
        
        
        cwiczenia2_menu = QMenu("Ćwiczenia 2", self)
        histogram = QAction("Histogram", self)
        spl_roz = QAction("Splot, rozmywanie", self)
        cwiczenia2_menu.addAction(histogram)
        cwiczenia2_menu.addAction(spl_roz)

        self.button2 = QPushButton("Ćwiczenie 2")
        self.button2.setMenu(cwiczenia2_menu)
        self.side_panel_layout.addWidget(self.button2)

        histogram.triggered.connect(self.histogram_menu_triggered)

        cwiczenia3_menu = QMenu("Ćwiczenia 3", self)
        self.button3 = QPushButton("Ćwiczenie 3")
        self.button3.setMenu(cwiczenia3_menu)
        self.side_panel_layout.addWidget(self.button3)

        cwiczenia4_menu = QMenu("Ćwiczenia 4", self)
        self.button4 = QPushButton("Ćwiczenie 4")
        self.button4.setMenu(cwiczenia4_menu)
        self.side_panel_layout.addWidget(self.button4)

        cwiczenia5_menu = QMenu("Ćwiczenia 5-6", self)
        self.button5 = QPushButton("Ćwiczenie 5-6")
        self.button5.setMenu(cwiczenia5_menu)
        self.side_panel_layout.addWidget(self.button5)

        cwiczenia7_menu = QMenu("Ćwiczenia 7", self)
        self.button7 = QPushButton("Ćwiczenie 7")
        self.button7.setMenu(cwiczenia7_menu)
        self.side_panel_layout.addWidget(self.button7)

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

        # Widget z suwakami do rozmywania
        self.blur_sliders_widget = QWidget()
        self.blur_sliders_layout = QVBoxLayout()
        self.blur_sliders_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.blur_sliders_widget.setLayout(self.blur_sliders_layout)
        self.blur_sliders_widget.setVisible(False)  # Początkowo niewidoczny

        self.blur_sliders = {}
        blur_names = {
            "splot": "splot",
            "równ. rozmywanie": "rozmazywanie",
            "gauss. rozmywanie": "gauss",
            "splot z maską": "splot_maska",
        }

        for name, function_name in blur_names.items():
            label = QLabel(f"{name}: 0")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, l=label, n=name: l.setText(f"{n}: {value}"))
            slider.valueChanged.connect(lambda value, fn=function_name: getattr(self, fn)(value))
            self.blur_sliders_layout.addWidget(label)
            self.blur_sliders_layout.addWidget(slider)
            self.blur_sliders[name] = slider

        # Dodajemy widgety z suwakami do panelu bocznego
        self.side_panel_layout.addWidget(self.blur_sliders_widget)

        spl_roz.triggered.connect(self.blur_menu_triggered)


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
        index = self.side_panel_layout.indexOf(self.button2)
        self.side_panel_layout.insertWidget(index, self.lut_sliders_widget)

        edge_roberts = QAction("Detekcja krawędzi: Roberts", self)
        edge_prewitt = QAction("Detekcja krawędzi: Prewitt", self)
        edge_sobel = QAction("Detekcja krawędzi: Sobel", self)
        edge_laplacian = QAction("Detekcja krawędzi: Laplace", self)

        cwiczenia3_menu.addAction(edge_roberts)
        cwiczenia3_menu.addAction(edge_prewitt)
        cwiczenia3_menu.addAction(edge_sobel)
        cwiczenia3_menu.addAction(edge_laplacian)
        edge_roberts.triggered.connect(lambda: self.apply_edge_filter("roberts"))
        edge_prewitt.triggered.connect(lambda: self.apply_edge_filter("prewitt"))
        edge_sobel.triggered.connect(lambda: self.apply_edge_filter("sobel"))
        edge_laplacian.triggered.connect(lambda: self.apply_edge_filter("laplacian"))

        edge_log_zero = QAction("Detekcja krawędzi: Laplacian of Gauss (LoG)", self)
        cwiczenia3_menu.addAction(edge_log_zero)
        edge_log_zero.triggered.connect(self.apply_laplacian_of_gauss)

        edge_canny = QAction("Detekcja krawędzi: Canny", self)
        cwiczenia3_menu.addAction(edge_canny)
        edge_canny.triggered.connect(self.apply_edge_canny)

        bin_manual_action = QAction("Binaryzacja: ręczna", self)
        bin_otsu_action = QAction("Binaryzacja: Otsu", self)

        cwiczenia4_menu.addAction(bin_manual_action)
        cwiczenia4_menu.addAction(bin_otsu_action)

        bin_manual_action.triggered.connect(self.binaryzacja_manual)
        bin_otsu_action.triggered.connect(self.binaryzacja_otsu)

        # Widget binaryzacji ręcznej (suwak progu)
        self.binary_slider_widget = QWidget()
        self.binary_slider_layout = QVBoxLayout()
        self.binary_slider_widget.setLayout(self.binary_slider_layout)
        self.binary_slider_widget.setVisible(False)

        self.binary_slider = QSlider(Qt.Orientation.Horizontal)
        self.binary_slider.setMinimum(0)
        self.binary_slider.setMaximum(255)
        self.binary_slider.setValue(150)

        self.binary_slider_label = QLabel(f"Próg binaryzacji: {self.binary_slider.value()}")
        self.binary_slider.valueChanged.connect(self.update_binary_threshold)

        self.binary_slider_layout.addWidget(self.binary_slider_label)
        self.binary_slider_layout.addWidget(self.binary_slider)

        index_cwiczenie4 = self.side_panel_layout.indexOf(self.button4)
        self.side_panel_layout.insertWidget(index_cwiczenie4 + 1, self.binary_slider_widget)

        bin_niblack_action = QAction("Binaryzacja: Niblack", self)
        cwiczenia4_menu.addAction(bin_niblack_action)
        bin_niblack_action.triggered.connect(self.binaryzacja_niblack)

        hough_action = QAction("Transformata Hougha: Linie", self)
        cwiczenia5_menu.addAction(hough_action)
        hough_action.triggered.connect(self.apply_hough_transform)

        watershed_action = QAction("Segmentacja wododziałowa (Vincent-Soille)", self)
        cwiczenia5_menu.addAction(watershed_action)
        watershed_action.triggered.connect(self.apply_watershed)

        harris_action = QAction("Detekcja narożników: Harris", self)
        cwiczenia7_menu.addAction(harris_action)
        harris_action.triggered.connect(self.apply_harris_corner_detection)

    def apply_harris_corner_detection(self):
        if hasattr(self, 'image'):
            from corner_harris import CornerHarris
            detector = CornerHarris(
                self.image,
                sigma=1.0,
                sigma_weight=0.76,
                k_param=0.05,
                threshold=30000000
            )
            self.image = detector.transform()
            self.current_image = self.image.copy()
            self.display_image()

    def apply_watershed(self):
        if hasattr(self, 'image'):
            from segmentation import WatershedVincentSoille
            watershed = WatershedVincentSoille(self.image)
            self.image = watershed.transform()
            self.current_image = self.image.copy()
            self.display_image()

    def apply_hough_transform(self):
        if hasattr(self, 'image'):
            from hough_lines import HoughTransform
            hough = HoughTransform(self.image, theta_density=3.0, skip_edge_detection=False)
            self.image = hough.transform()
            self.current_image = self.image.copy()
            self.display_image()

    def binaryzacja_niblack(self):
        if hasattr(self, 'image'):
            from bin_niblack import BinNiblack
            b = BinNiblack(self.image, radius=3, k=-0.2)
            self.image = b.transform()
            self.current_image = self.image.copy()
            self.display_image()

    def binaryzacja_manual(self):
        if hasattr(self, 'image'):
            visible = self.binary_slider_widget.isVisible()
            self.binary_slider_widget.setVisible(not visible)

            if not visible:
                self.apply_binary_manual()

    def apply_binary_manual(self):
        from bin_manual import BinManual
        threshold = self.binary_slider.value()
        if hasattr(self, 'original_image'):
            # Operuj na obrazie wejściowym
            b = BinManual(self.original_image.copy(), threshold=threshold)
            self.image = b.transform()
            self.current_image = self.image.copy()
            self.display_image()

    def update_binary_threshold(self, value):
        self.binary_slider_label.setText(f"Próg binaryzacji: {value}")
        if not self.ignoring_binary_update:
            self.apply_binary_manual()

    def binaryzacja_otsu(self):
        if hasattr(self, 'image'):
            from bin_otsu import BinOtsu
            b = BinOtsu(self.image)
            self.image = b.transform()
            self.current_image = self.image.copy()
            self.display_image()


    def apply_edge_canny(self):
        if hasattr(self, 'image'):
            from edge_canny import EdgeCanny
            canny = EdgeCanny(self.image, lower_thresh=50, upper_thresh=100)
            self.image = canny.transform().convert("RGB")
            self.current_image = self.image.copy()
            self.display_image()

    def apply_laplacian_of_gauss(self):
        if hasattr(self, 'image'):
            log = EdgeLaplaceOfGauss(self.image, size=3, sigma=1.6, threshold=5)
            self.image = log.transform().convert("RGB")  # konwersja dla spójności
            self.current_image = self.image.copy()
            self.display_image()

    def apply_edge_filter(self, method):
        if not hasattr(self, 'image'):
            return

        if method == "roberts":
            edge = EdgeRoberts(self.image)
        elif method == "prewitt":
            edge = EdgePrewitt(self.image)
        elif method == "sobel":
            edge = EdgeSobel(self.image)
        elif method == "laplacian":
            edge = EdgeLaplacian(self.image, size=3)
        else:
            return

        self.image = edge.transform()
        self.current_image = self.image.copy()
        self.display_image()

    def blur_menu_triggered(self):
        self.blur_sliders_widget.setVisible(not self.blur_sliders_widget.isVisible())

    def validate_histogram_checkboxes(self, changed_channel, checked):
        if not checked:
            # Liczymy ile jest innych zaznaczonych checkboxów
            remaining_checked = sum(
                cb.isChecked() for name, cb in self.hist_checkboxes.items() if name != changed_channel
            )
            if remaining_checked == 0:
                # Jeśli to ostatni, cofamy zmianę
                self.hist_checkboxes[changed_channel].setChecked(True)

    def histogram_menu_triggered(self):
        if hasattr(self, 'image'):
            dialog = HistogramDialog(self.image, self)
            dialog.image_modified.connect(self.update_main_image)
            dialog.exec()

    def update_main_image(self, new_pil_image):
        self.image = new_pil_image
        self.current_image = self.image.copy()
        self.display_image()

        
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
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*);;PPM Files (*.ppm)")
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
        pixels = self.image.load()
        for i in range(self.image.width):
            for j in range(self.image.height):
                r, g, b = pixels[i, j]
                pixels[i, j] = (255 - r, 255 - g, 255 - b)

    def display_image(self):
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

        for slider_info in [self.brightness_slider, self.contrast_slider, self.gamma_slider]:
            slider_info["slider"].valueChanged.connect(self.apply_lut_correction)

    def apply_lut_correction(self):
        if hasattr(self, 'original_image'):
            shift = self.brightness_slider["slider"].value()
            contrast_val = self.contrast_slider["slider"].value()
            gamma_val = self.gamma_slider["slider"].value()

            factor = 1.0 + (contrast_val / 100.0)
            gamma = 1.0 + (gamma_val / 100.0)

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

            self.ignoring_binary_update = True
            self.binary_slider.setValue(0)
            self.ignoring_binary_update = False

    def splot(self, value):
        if self.original_image is None:
            return

        img_array = np.array(self.original_image)

        radius = max(1, int(value / 10)) if value > 0 else 0

        if radius == 0:
            self.image = self.original_image.copy()
            self.current_image = self.image.copy()
            self.display_image()
            return

        kernel_size = 2 * radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= kernel.sum()
        border_type = cv2.BORDER_REPLICATE

        if img_array.ndim == 3 and img_array.shape[2] == 3:
            convolved = np.zeros_like(img_array)
            for i in range(3):
                convolved[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel, borderType=border_type)
        else:
            convolved = cv2.filter2D(img_array, -1, kernel, borderType=border_type)

        convolved_img = Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8))

        self.image = convolved_img
        self.current_image = self.image.copy()
        self.display_image()

    def rozmazywanie(self, value):
        if self.original_image is None:
            return

        img_array = np.array(self.original_image)

        radius = max(1, int(value / 10)) if value > 0 else 0

        if radius == 0:
            self.image = self.original_image.copy()
            self.current_image = self.image.copy()
            self.display_image()
            return

        kernel_size = 2 * radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= kernel.sum()

        # Używamy domyślnego typu brzegowego
        convolved = cv2.filter2D(img_array, -1, kernel)

        convolved_img = Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8))
        self.image = convolved_img
        self.current_image = self.image.copy()
        self.display_image()

    def gauss(self, value):
        if self.original_image is None:
            return

        img_array = np.array(self.original_image)

        radius = max(1, int(value / 10))
        if radius == 0:
             radius = 1

        sigma = 0.1 + (value / 100.0) * 4.9

        if value == 0:
            self.image = self.original_image.copy()
            self.current_image = self.image.copy()
            self.display_image()
            return

        kernel_size = 2 * radius + 1
        gaussian_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

        center = radius

        for i in range(kernel_size):
            for j in range(kernel_size):
                x = j - center
                y = i - center
                
                if sigma <= 0:
                    raise ValueError("Sigma musi być większa od zera.")
                term1 = 1 / (2 * np.pi * sigma**2)
                term2 = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                gaussian_kernel[i, j] = term1 * term2

        if gaussian_kernel.sum() != 0:
            gaussian_kernel /= gaussian_kernel.sum()
        else:
            print("Ostrzeżenie: Suma maski Gaussa wynosi zero.")

        border_type = cv2.BORDER_REPLICATE

        if img_array.ndim == 3 and img_array.shape[2] == 3:
            convolved = np.zeros_like(img_array, dtype=np.float32)
            for i in range(3):
                convolved[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, gaussian_kernel, borderType=border_type)
        else:
            convolved = cv2.filter2D(img_array, -1, gaussian_kernel, borderType=border_type)

        convolved_img = Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8))

        self.image = convolved_img
        self.current_image = self.image.copy()
        self.display_image()

    def splot_maska(self, value):
        if self.original_image is None:
            return
        img_array = np.array(self.original_image)
        border_type = cv2.BORDER_REPLICATE
        if value == 0:
            self.image = self.original_image.copy()
            self.current_image = self.image.copy()
            self.display_image()
            return
        intensity = value / 100.0
        kernel = np.array([[-intensity, -intensity, -intensity],
                           [-intensity, 1 + 8 * intensity, -intensity],
                           [-intensity, -intensity, -intensity]], dtype=np.float32)
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            convolved = np.zeros_like(img_array, dtype=np.float32)
            for i in range(3):
                convolved[:, :, i] = cv2.filter2D(img_array[:, :, i], -1, kernel, borderType=border_type)
        else:
            convolved = cv2.filter2D(img_array, -1, kernel, borderType=border_type)
        convolved_img = Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8))
        self.image = convolved_img
        self.current_image = self.image.copy()
        self.display_image()
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())