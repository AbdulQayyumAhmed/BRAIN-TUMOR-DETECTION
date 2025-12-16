# 🧠 Brain Tumor Detection System (AI Powered)

This project is a **Brain Tumor Detection System** built using **Deep Learning** and **Transfer Learning** with **CNN models**.  
The system can classify **Brain MRI images** as **Tumor** or **No Tumor**.  
It also has a **Streamlit-based interface** for easy interaction and is ready for deployment on **Streamlit Cloud**.

---

## Features

- Upload MRI images via a **web interface**.
- Detects presence of a **brain tumor**.
- Shows **confidence score (%)**.
- Uses a **pre-trained MobileNetV2** model with **transfer learning**.
- Optional: fine-tuning for improved accuracy.
- Fully deployable on **Streamlit Cloud**.

---

## Dataset

- **Dataset used**: Brain Tumor MRI Dataset (Open-source, Kaggle/Figshare)
- **Classes**: 
  - Tumor
  - No Tumor
- Dataset folder structure:

dataset/
├── tumor/
│ ├── img1.jpg
│ └── ...
└── no_tumor/
├── img1.jpg
└── ...


References

Kaggle: Brain Tumor MRI Dataset

MobileNetV2 Paper

Streamlit Documentation