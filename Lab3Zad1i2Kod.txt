from scipy.stats import norm
from csv import writer
import numpy as np
import random
from sklearn.cluster import KMeans
from pyransac3d import Plane
import os
import cv2

def process_textures(input_dir, output_dir):
    numer_obrazu = 1

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Nie udało się wczytać obrazu: {filename}")
                continue

            height, width, _ = img.shape
            probka_height = 128
            probka_width = 128


            for y in range(0, height - probka_height + 1, probka_height):
                for x in range(0, width - probka_width + 1, probka_width):
                    wycinek = img[y:y + probka_height, x:x + probka_width]
                    wycinek_filename = os.path.join(output_dir, f"{numer_obrazu}.png")
                    cv2.imwrite(wycinek_filename, wycinek)
                    numer_obrazu += 1

kategorie = {
    'Floor': 'FloorProcesed',
    'Wall': 'WallProcesed',
    'Furniture': 'FurnitureProcesed',
}

for input_folder, output_folder in kategorie.items():
    process_textures(input_dir=input_folder, output_dir=output_folder)