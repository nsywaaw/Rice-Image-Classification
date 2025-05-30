import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# ------------------- Preprocessing & Feature Extraction -------------------

def preprocess_image_cv2(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    for i in range(7):
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i]) + 1e-10)
    return huMoments.flatten()

# ------------------- Load Model -------------------

try:
    with open('knn_model.pkl', 'rb') as f:
        knn_model, categories = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'knn_model.pkl' tidak ditemukan. Jalankan train_model.py terlebih dahulu.")
    st.stop()

# ------------------- Streamlit UI -------------------

st.title("Prediksi Varietas Beras dari Citra")
st.write("Upload gambar beras, dan sistem akan memprediksi varietasnya.")

uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Gambar Beras", use_container_width=True)

    # Preprocessing
    processed_img = preprocess_image_cv2(img_array)

    # Ekstraksi fitur
    glcm = extract_glcm_features(processed_img)
    hu = extract_hu_moments(processed_img)

    # Gabungkan fitur jadi satu array
    feature_vector = np.concatenate([
        glcm['contrast'], glcm['correlation'],
        glcm['energy'], glcm['homogeneity'],
        hu
    ]).reshape(1, -1)

    # Prediksi
    prediction = knn_model.predict(feature_vector)[0]
    predicted_class = categories[prediction]

    st.success(f"Prediksi varietas: **{predicted_class}**")
