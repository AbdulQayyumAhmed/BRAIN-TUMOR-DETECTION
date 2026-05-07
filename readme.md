# 🧠 NeuroScan AI | Brain Tumor Detection System

**NeuroScan AI** is a high-performance, deep-learning powered diagnostic dashboard designed to assist medical professionals in the rapid analysis of Brain MRI scans.

---

## ✨ Features

- **Modern Single-Page Dashboard**: A premium, wide-layout interface that fits perfectly on one screen.
- **AI-Powered Detection**: Leverages a Deep Convolutional Neural Network (CNN) to detect tumors with high precision.
- **Glassmorphism Aesthetic**: Sleek medical-themed UI with semi-transparent, blurred elements.
- **Real-Time Analysis**: Instant results with confidence scores displayed directly below the analysis trigger.
- **Diagnostic Viewport**: High-resolution image preview with strict vertical constraints to maintain layout stability.
- **Cloud Ready**: Optimized for deployment on Streamlit Cloud or Docker environments.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.x, TensorFlow/Keras
- **Frontend**: Streamlit (with Custom CSS Injection)
- **Image Processing**: PIL (Pillow), NumPy
- **Model**: Custom CNN (Informed by MobileNetV2 architecture)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

### Running Locally
```powershell
streamlit run app.py
```

---

## 📂 Project Structure

- `app.py`: The core Streamlit application logic and UI.
- `brain_tumor_model.keras`: The pre-trained deep learning model.
- `train_model.py`: Script used for training/fine-tuning the detection model.
- `dataset/`: Training and validation data directory.

---

## ⚠️ Medical Disclaimer
This tool is intended for **research and educational purposes only**. It should not be used as a replacement for professional medical diagnosis. Always consult a certified radiologist or healthcare provider for clinical evaluations.

---

*Developed with ❤️ by NeuroScan AI Team*