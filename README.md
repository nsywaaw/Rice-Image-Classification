# Rice Variety Classifier 

A Streamlit-based web application that predicts the **variety of rice grain** from an uploaded image using a trained **K-Nearest Neighbors (KNN)** model and **GLCM + Hu Moments feature extraction**.

---

## Features

- Upload an image of a single rice grain.
- The system classifies it into one of the following varieties:
  - Arborio
  - Basmati
  - Ipsala
  - Jasmine
  - Karacadag
- Image preprocessing and feature extraction using:
  - CLAHE, Median + Gaussian Blur
  - GLCM Texture Features
  - Hu Moments
    
---

## Project Structure
  - app.py 
  - knn_model.pkl 
  - requirements.txt 
  - train_model.py 
  - README.md 
