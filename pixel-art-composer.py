import cv2
import numpy as np
from numba import jit
import sys

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
    if scale <= 0.0:
        raise ValueError("Scale must be a positive number")
    resized = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized

def main(image_path, hex_color_file_path, pixelation_scale, output_file_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read the image: {image_path}")
        return 1

    palette = []
    with open(hex_color_file_path, 'r') as file:
        for line in file:
            palette.append(hex_to_BGR(line.strip()))

    palette = np.array(palette, dtype=np.uint8)

    scale = float(pixelation_scale)
    resized_image = resize_image(image, scale)
    processed_image = process_image(resized_image, palette)

    if not cv2.imwrite(output_file_path, processed_image):
        print("Failed to save the modified image.")
        return 1

    print(f"The image has been converted to use the indexed colors from the HEX file. Result saved as {output_file_path}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <ImagePath> <HexColorFilePath> <PixelationScale> <OutputFilePath>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
