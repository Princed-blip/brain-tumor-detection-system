# NeuroScan AI - Brain Tumor Detection System

An AI-powered brain tumor detection and classification system built
using Deep Learning and Medical Imaging as a Final Year Project.

## Overview
This system classifies brain MRI scans into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

## Model
- **Architecture:** EfficientNetV2-B0 with Transfer Learning
- **Strategy:** Two-phase training (head training + fine-tuning)
- **Explainability:** Grad-CAM heatmap visualization
- **Parameters:** ~7.1 Million
- **Framework:** TensorFlow 2.x / Keras

## Results
| Metric | Value |
|--------|-------|
| Test Accuracy | 71.57% |
| Validation Accuracy | 82.20% |
| AUC-ROC | 0.8928 |
| No Tumor Sensitivity | 90.48% |
| Glioma Sensitivity | 39.00% |

## Model Comparison
| Metric | EfficientNetV2 | ResNet50 |
|--------|---------------|----------|
| Test Accuracy | 71.57% | 76.90% |
| Val Accuracy | 82.20% | 89.70% |
| AUC-ROC | 0.8928 | 0.9392 |
| Parameters | ~7.1M | ~25.6M |
| Glioma Sensitivity | 39.00% | 29.00% |

## How to Run

### 1. Clone the repository
    git clone https://github.com/YOUR_USERNAME/neuroscan-ai.git
    cd neuroscan-ai

### 2. Install dependencies
    pip install -r requirements.txt

### 3. Download the dataset
Download the Kaggle Brain MRI Dataset from:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
and place it in the archive/ folder.

### 4. Run the web app
    streamlit run streamlit_app.py

## Project Structure
    neuroscan-ai/
    ├── streamlit_app.py           # Web application
    ├── BTP.ipynb                  # Training notebook
    ├── best_model_v2_phase2.keras # Trained model weights
    ├── requirements.txt           # Dependencies
    ├── static/uploads/            # Temporary upload folder
    └── archive/                   # Dataset (not included)
        ├── Training/
        └── Testing/

## Features
- Multi-class brain tumor classification
- Grad-CAM visual explainability
- Per-class Sensitivity & Specificity reporting
- EfficientNetV2 vs ResNet50 comparison
- Interactive Streamlit web interface

## Disclaimer
This system is a research prototype for academic purposes only.
It is not intended for clinical use and must not replace
professional medical diagnosis.

## Developer
- **Name:** Stephen Ifeanyi
- **Brand:** Princedex
- **Email:** ifeanyistephen003@gmail.com

## Dataset
Chakrabarty, N. (2021). Brain MRI Images for Brain Tumor Detection.
Kaggle. https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri