import sys
import os
import tempfile
import cv2
import numpy as np
from numba import jit
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QFileDialog,
                             QLineEdit, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QDoubleValidator


os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'pixel-art-composer/lib/python3.10/site-packages/cv2/qt/plugins/platforms'

def get_application_path():
    if getattr(sys, 'frozen', False):
        # The application is running as a bundled executable
        application_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(sys.executable)))
    else:
        # The application is running as a script in a development environment
        application_path = os.path.dirname(os.path.abspath(__file__))
    return application_path

def hex_to_BGR(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], dtype=np.uint8)

@jit(nopython=True)
def find_closest_palette_color(color, palette):
    min_distance = np.inf
    index = -1
    for i in range(palette.shape[0]):
        distance = np.sum((color - palette[i]) ** 2)
        if distance < min_distance:
            min_distance = distance
            index = i
    return index

@jit(nopython=True)
def process_image(image, palette):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i, j]
            closest_color_index = find_closest_palette_color(pixel, palette)
            image[i, j] = palette[closest_color_index]
    return image

def resize_image(image, scale):
    if scale < 0.0:
        raise ValueError("Scale must be a positive number")
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized

def all_procss(image_path, hex_color_file_path, pixelation_scale, output_file_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image: {image_path}")
        return 1

    palette = []
    with open(hex_color_file_path, 'r') as file:
        for line in file:
            palette.append(hex_to_BGR(line.strip()))

    print("Opening hex color file path:", hex_color_file_path)
    if os.path.isdir(hex_color_file_path):
        print(f"Expected a file but got a directory: {hex_color_file_path}")
        return 1

    palette = np.array(palette, dtype=np.uint8)

    scale = float(pixelation_scale)
    resized_image = resize_image(image, scale)
    processed_image = process_image(resized_image, palette)

    if not cv2.imwrite(output_file_path, processed_image):
        print("Failed to save the modified image.")
        return 1

    print(f"The image has been converted to use the indexed colors from the HEX file. Result saved as {output_file_path}")
    return 0

class PixelArtComposer(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.loaded_image_path = ""
        self.pixelation_scale_slider = None
        self.pixelation_scale_input = None
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.setupImageViewer()
        self.setupImageLoader()
        self.setupHexDropdown()
        self.setupPixelArtControls()
        self.setupExecuteButton()
        self.setupSaveButton()
        self.finalizeLayout()

    def setupImageViewer(self):
        self.scrollArea = QScrollArea(self)
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setWidgetResizable(True)
        self.layout.addWidget(self.scrollArea)

    def setupImageLoader(self):
        self.btnLoadImage = QPushButton('Load Image', self)
        self.btnLoadImage.clicked.connect(self.openImageDialog)
        self.layout.addWidget(self.btnLoadImage)

    def setupHexDropdown(self):
        self.hexLayout = QHBoxLayout()

        self.btnLoadHexFiles = QPushButton('Load Hex Folder', self)
        self.btnLoadHexFiles.clicked.connect(self.openHexFolderDialog)
        self.hexLayout.addWidget(self.btnLoadHexFiles)

        self.hexDropdown = QComboBox(self)
        self.hexDropdown.hide() 
        self.hexLayout.addWidget(self.hexDropdown)

        self.layout.addLayout(self.hexLayout)

    def openHexFolderDialog(self):
        options = QFileDialog.Options()
        folderPath = QFileDialog.getExistingDirectory(self, "Select Hex Folder", options=options)
        if folderPath:
            self.loadHexFiles(folderPath)

    def loadHexFiles(self, folderPath):
        hex_files = [f for f in os.listdir(folderPath) if f.endswith('.hex') or f.endswith('.txt')]
        if hex_files:
            self.hexDropdown.clear()  # Clear existing items before adding new ones
            self.hexDropdown.addItems(hex_files)

            self.btnLoadHexFiles.hide()
            self.hexDropdown.show()
        else:
            print("No .hex or .txt files found in the selected folder.")


    def createParameterControl(self, label_text):
        mainWidget = QWidget()
        mainLayout = QVBoxLayout(mainWidget)

        label = QLabel(label_text)
        mainLayout.addWidget(label)

        sliderLayout = QHBoxLayout()
        inputField = QLineEdit(self)

        slider = QSlider(Qt.Horizontal, self)

        if label_text == "Pixelation Scale:":
            self.pixelation_scale_input = inputField
            inputField.setValidator(QDoubleValidator(0.0, 1.0, 2))
            slider.setRange(0, 100)
            slider.setValue(50)
            inputField.setText("0.5")
            slider.valueChanged.connect(lambda value: inputField.setText(str(value / 100.0)))
            inputField.editingFinished.connect(lambda: slider.setValue(int(float(inputField.text()) * 100)))
        sliderLayout.addWidget(inputField)
        sliderLayout.addWidget(slider)

        mainLayout.addLayout(sliderLayout)
        mainLayout.setSpacing(5)
        return mainWidget
    
    def setupPixelArtControls(self):
        pixelationControl = self.createParameterControl("Pixelation Scale:")
        self.layout.addWidget(pixelationControl)

    def openImageDialog(self):
        options = QFileDialog.Options()
        imagePath, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)",
                                                   options=options)
        if imagePath:
            self.loaded_image_path = imagePath
            self.loadImage(imagePath)

    def loadImage(self, path):
        self.image = QPixmap(path)
        self.updateImageDisplay()

    def updateImageDisplay(self):
        if self.image is not None:
            scaledImage = self.image.scaled(self.scrollArea.size(), Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(scaledImage)

    def executePixelArtComposer(self):
        if self.image is None:
            print("No image loaded.")
            return

        _, input_image_extension = os.path.splitext(self.loaded_image_path)

        output_image = os.path.join(tempfile.gettempdir(), f'output_pixel_art{input_image_extension}')
        print("Selected hex file from dropdown:", self.hexDropdown.currentText())
        palette_file = os.path.join(get_application_path(), 'hex', self.hexDropdown.currentText())
        print("Constructed palette_file path:", palette_file)

        if self.pixelation_scale_input is None:
            print("Pixelation scale input is not initialized.")
            return

        pixelation_scale = float(self.pixelation_scale_input.text())
        input_image_path = os.path.abspath(str(self.loaded_image_path))
        all_procss(input_image_path, palette_file, str(pixelation_scale), output_image)

        self.loadImage(output_image)

    def setupExecuteButton(self):
        self.btnExecute = QPushButton('Compose pixel art', self)
        self.btnExecute.clicked.connect(self.executePixelArtComposer)
        self.layout.addWidget(self.btnExecute)

    def setupSaveButton(self):
        self.btnSave = QPushButton('Save image', self)
        self.btnSave.clicked.connect(self.saveImageDialog)
        self.layout.addWidget(self.btnSave)

    def saveImageDialog(self):
        options = QFileDialog.Options()
        savePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)",
                                                  options=options)
        if savePath:
            self.saveImage(savePath)

    def saveImage(self, path):
        if self.imageLabel.pixmap():
            self.imageLabel.pixmap().save(path)
        else:
            print("No image to save.")
    
    def finalizeLayout(self):
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setWindowTitle('Pixel art composer GUI')
        self.setGeometry(300, 300, 600, 400)

def main():
    app = QApplication(sys.argv)
    ex = PixelArtComposer()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

