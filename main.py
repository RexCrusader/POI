import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

def reduce_gray_levels(obraz, levels=64):
    factor = 256 // levels
    return (obraz // factor).astype(np.uint8)

def extract_features(image, distances, angles, levels=64):
    glcm = graycomatrix(image,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)

    features = {}
    props = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
    for prop in props:
        result = graycoprops(glcm, prop)
        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                angle_deg = int(np.rad2deg(a))
                features[f"{prop}_d{d}_a{angle_deg}"] = result[i, j]

    return features

def process_category(category_folder, label):
    data = []
    for filename in os.listdir(category_folder):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(category_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_reduced = reduce_gray_levels(gray, levels=64)

            features = extract_features(
                gray_reduced,
                distances=[1, 3, 5],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=64
            )
            features['label'] = label
            data.append(features)
    return data

all_data = []
all_data += process_category("FloorProcesed", "Floor")
all_data += process_category("WallProcesed", "Wall")
all_data += process_category("FurnitureProcesed", "Furniture")

df = pd.DataFrame(all_data)
df.to_csv("cechy_tekstur.csv", index=False)
print("Zapisano cechy do pliku cechy_tekstur.csv")