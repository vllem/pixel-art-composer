import sys
import os
import tempfile
import cv2
import numpy as np
from numba import jit
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QFileDialog,
                             QLineEdit, QComboBox, QScrollArea, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QDoubleValidator
from scipy.signal import convolve2d

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'pixel-art-composer/lib/python3.10/site-packages/cv2/qt/plugins/platforms'

def get_resource_path(relative_path):
    base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def hex_to_BGR(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return np.array([b, g, r], dtype=np.uint8)

def ensure_odd(number):
    number = abs(int(number)) 
    return number if number % 2 == 1 else number + 1

def add_black_edges(input_image, low_threshold, high_threshold, blur_kernel_size, edge_color, line_thickness):
    image = input_image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blur_kernel_size = ensure_odd(blur_kernel_size)
    blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    edges = cv2.Canny(blur, low_threshold, high_threshold)

    
    edge_color = hex_to_BGR(edge_color)

    if line_thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, line_thickness))
        edges = cv2.dilate(edges, kernel)

    transparent_edges = np.zeros((edges.shape[0], edges.shape[1], 4), dtype=np.uint8)

    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x]:
                transparent_edges[y, x] = [edge_color[0], edge_color[1], edge_color[2], 255]

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if transparent_edges[y, x][3] > 0:
                image[y, x] = [transparent_edges[y, x][0], transparent_edges[y, x][1], transparent_edges[y, x][2]]

    return image

# @jit(nopython=True)
# def process_image(image, palette):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel = image[i, j]
#             closest_color_index = find_closest_palette_color(pixel, palette)
#             image[i, j] = palette[closest_color_index]
#     return image

@jit(nopython=True)
def process_image(image, palette, find_closest=True):
    height, width, _ = image.shape
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]
            best_color_index = -1
            if find_closest:
                best_distance = np.inf
            else:
                best_distance = -np.inf

            for k in range(palette.shape[0]):
                distance = np.sum((pixel - palette[k]) ** 2)
                
                if find_closest and distance < best_distance:
                    best_distance = distance
                    best_color_index = k
                elif not find_closest and distance > best_distance:
                    best_distance = distance
                    best_color_index = k

            if best_color_index == -1:
                raise ValueError("No best color index found.")

            image[i, j] = palette[best_color_index]

    return image

# @jit(nopython=True)
# def custom_gaussian_kernel(size, sigma=1.0):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#     return g / g.sum()

# @jit(nopython=True)
# def custom_gaussian_filter(image, sigma=1.0):
#     kernel_size = 2 * int(4 * sigma + 0.5) + 1
#     kernel = custom_gaussian_kernel(kernel_size, sigma)
#     filtered_image = np.zeros_like(image)
#     for i in range(3):
#         filtered_image[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same', boundary='fill', fillvalue=0)
#     return filtered_image

@jit(nopython=True)
def process_image_blocks(image, palette, find_closest=True, block_size_width=2, block_size_height=2):
    height, width, _ = image.shape
    processed_image = np.zeros_like(image)
    for i in range(0, height, block_size_height):
        for j in range(0, width, block_size_width):
            block = image[i:i+block_size_height, j:j+block_size_width]
            
            total_color = np.zeros(3)
            count = 0
            for bi in range(block.shape[0]):
                for bj in range(block.shape[1]):
                    total_color += block[bi, bj, :]
                    count += 1
            mean_color = total_color / count if count > 0 else np.zeros(3)
            
            best_color_index = -1
            if find_closest:
                best_distance = np.inf
            else:
                best_distance = -np.inf
            for k in range(palette.shape[0]):
                distance = np.sum((mean_color - palette[k]) ** 2)
                if find_closest and distance < best_distance:
                    best_distance = distance
                    best_color_index = k
                elif not find_closest and distance > best_distance:
                    best_distance = distance
                    best_color_index = k

            if best_color_index == -1:
                raise ValueError("No best color index found.")

            processed_image[i:i+block_size_height, j:j+block_size_width] = palette[best_color_index]
    
    return processed_image


def resize_image(image, scale):
    if scale < 0.0:
        raise ValueError("Scale must be a positive number")
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized

def all_process(image, hex_color_file_path, pixelation_scale, output_file_path, find_closest, block, block_size_width, block_size_height):
    if image is None or image.size == 0:
        print("Error: Image is empty or not loaded correctly.")
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
    if not 0.0 < scale <= 1.0:
        print("Error: Scale must be between 0.0 and 1.0")
        return 1
    
    resized_image = resize_image(image, scale)
    if resized_image is None or resized_image.size == 0:
        print("Error: Resized image is empty.")
        return 1
    
    if block == False:
        if find_closest == True:
            processed_image = process_image(resized_image, palette, True)
        else:
            processed_image = process_image(resized_image, palette, False)
    else:
        if find_closest == True:
            processed_image = process_image_blocks(resized_image, palette, True, block_size_width, block_size_height)
        else:
            processed_image = process_image_blocks(resized_image, palette, False, block_size_width, block_size_height)

    if processed_image is None or processed_image.size == 0:
        print("Error: Processed image is empty.")
        return 1

    if not cv2.imwrite(output_file_path, processed_image):
        print("Failed to save the modified image.")
        return 1

    print(f"The image has been converted to use the indexed colors from the HEX file. Result saved as {output_file_path}")
    return 0

class PixelArtComposer(QWidget):
    def __init__(self, temp_dir_path):
        super().__init__()
        self.temp_dir_path = temp_dir_path
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
            self.hexDropdown.clear()
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
        
        return mainWidget, mainLayout
    
    def addLabeledInput(self, layout, label, defaultValue):
        hbox = QHBoxLayout()
        lbl = QLabel(label, self)
        inputField = QLineEdit(self)
        inputField.setText(defaultValue)
        hbox.addWidget(lbl)
        hbox.addWidget(inputField)
        layout.addLayout(hbox)
        return inputField
    
    def setupPixelArtControls(self):
        pixelationControl, pixelationControlLayout = self.createParameterControl("Pixelation Scale:")
        self.layout.addWidget(pixelationControl)

        firstRow = QHBoxLayout()
        
        applyEdgesLabel = QLabel("Apply Edges", self)
        self.applyEdgesCheckbox = QCheckBox("", self)
        self.applyEdgesCheckbox.setChecked(False)

        firstRow.addWidget(applyEdgesLabel)
        firstRow.addWidget(self.applyEdgesCheckbox)

        self.lowThresholdInput = self.addLabeledInput(firstRow, "Low Threshold:", "10")
        self.highThresholdInput = self.addLabeledInput(firstRow, "High Threshold:", "1000")
        self.blurKernelSizeInput = self.addLabeledInput(firstRow, "Blur Kernel Size:", "101")
        self.edgeColorInput = self.addLabeledInput(firstRow, "Edge Color (#RRGGBB):", "#000000")
        self.lineThicknessInput = self.addLabeledInput(firstRow, "Line Thickness:", "1")
    
        self.layout.addLayout(firstRow)
        
        secondRow = QHBoxLayout()

        furthestColorLabel = QLabel("Use Furthest Colors", self)
        self.furthestColorCheckbox = QCheckBox("", self)
        self.furthestColorCheckbox.setChecked(False) 

        secondRow.addWidget(furthestColorLabel)
        secondRow.addWidget(self.furthestColorCheckbox)

        processBlocksLabel = QLabel("Use Blocks Processing", self)
        self.processBlocksCheckbox = QCheckBox("", self)
        self.processBlocksCheckbox.setChecked(False)

        secondRow.addWidget(processBlocksLabel)
        secondRow.addWidget(self.processBlocksCheckbox)

        self.blockSizeHeightInput = self.addLabeledInput(secondRow, "Block Size Height:", "2")
        self.blockSizeWidthInput = self.addLabeledInput(secondRow, "Block Size Width:", "2")


        self.layout.addLayout(secondRow)


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

        output_image = os.path.join(self.temp_dir_path, f'output_pixel_art{input_image_extension}')    
        print("Selected hex file from dropdown:", self.hexDropdown.currentText())
        palette_file = get_resource_path(os.path.join('hex', self.hexDropdown.currentText()))
        print("Constructed palette_file path:", palette_file)

        if self.pixelation_scale_input is None:
            print("Pixelation scale input is not initialized.")
            return

        pixelation_scale = float(self.pixelation_scale_input.text())
        input_image_path = os.path.abspath(str(self.loaded_image_path))
        
        image = cv2.imread(input_image_path)
        if image is None:
            print(f"Could not read the image: {input_image_path}")
            return 1

        if image.size == 0:
            print(f"{input_image_path} image size is 0")
            return 1

        if self.applyEdgesCheckbox.isChecked():
            low_threshold = int(self.lowThresholdInput.text() or "10")
            high_threshold = int(self.highThresholdInput.text() or "100")
            blur_kernel_size = int(self.blurKernelSizeInput.text() or "3")
            edge_color = self.edgeColorInput.text() or "#000000"
            line_thickness = int(self.lineThicknessInput.text() or "1")

            image = add_black_edges(image, low_threshold, high_threshold, blur_kernel_size, edge_color, line_thickness)
        
        blockSizeHeight = int(self.blockSizeHeightInput.text() or "2")
        blockSizeWidth = int(self.blockSizeWidthInput.text() or "2")


        if self.processBlocksCheckbox.isChecked():

            if self.furthestColorCheckbox.isChecked():
                all_process(image, palette_file, str(pixelation_scale), output_image, False, True, blockSizeWidth, blockSizeHeight)
            else:
                all_process(image, palette_file, str(pixelation_scale), output_image, True, True, blockSizeWidth, blockSizeHeight)
        else:
            if self.furthestColorCheckbox.isChecked():
                all_process(image, palette_file, str(pixelation_scale), output_image, False, False, blockSizeWidth, blockSizeHeight)
            else:
                all_process(image, palette_file, str(pixelation_scale), output_image, True, False, blockSizeWidth, blockSizeHeight)


        self.loadImage(output_image)

    def setupExecuteButton(self):
        self.btnExecute = QPushButton('Compose Pixel Art', self)
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
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.setWindowTitle('Pixel Art Composer')
        self.setGeometry(600, 600, 1200, 800)

def main():
    app = QApplication(sys.argv)
    with tempfile.TemporaryDirectory() as temp_dir_path:
        ex = PixelArtComposer(temp_dir_path)
        ex.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()

