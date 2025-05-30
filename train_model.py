import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import graycomatrix, graycoprops

# --- PREPROCESSING & FEATURE EXTRACTION FUNCTIONS ---

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(gaussian, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_eq = clahe.apply(median)
    stretched = cv2.normalize(hist_eq, None, 0, 255, cv2.NORM_MINMAX)
    return stretched

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    features = {
        'contrast': graycoprops(glcm, 'contrast').flatten(),
        'correlation': graycoprops(glcm, 'correlation').flatten(),
        'energy': graycoprops(glcm, 'energy').flatten(),
        'homogeneity': graycoprops(glcm, 'homogeneity').flatten()
    }
    return features

def extract_hu_moments(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    huMoments = cv2.HuMoments(moments)
    for i in range(0,7):
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i])+1e-10)
    return huMoments.flatten()

# --- LOAD DATASET ---
base_path = 'Rice_Image_Dataset'
categories = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
print("Kategori ditemukan:", categories)

data = []
labels = []

for idx, category in enumerate(categories):
    folder_path = os.path.join(base_path, category)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        if os.path.isfile(img_path):
            try:
                processed = preprocess_image(img_path)
                glcm = extract_glcm_features(processed)
                hu = extract_hu_moments(processed)
                features = np.concatenate([
                    glcm['contrast'], glcm['correlation'],
                    glcm['energy'], glcm['homogeneity'],
                    hu
                ])
                data.append(features)
                labels.append(idx)
            except Exception as e:
                print(f"Gagal memproses {img_path}: {e}")

# --- TRAINING ---
X = np.array(data)
y = np.array(labels)

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- SAVE MODEL ---
with open('knn_model.pkl', 'wb') as f:
    pickle.dump((knn, categories), f)

print("Model berhasil disimpan ke knn_model.pkl")
