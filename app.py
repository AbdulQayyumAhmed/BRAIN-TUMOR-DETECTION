import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection System (AI Powered)")
st.write("Upload a Brain MRI image to detect whether a tumor is present.")

# -------------------------------
# Load Model (with caching)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Detect Tumor"):
    if image is None:
        st.warning("⚠️ Please upload an MRI image first.")
    else:
        # Preprocess image
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img)[0][0]

        # Confidence
        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"🧠 **Tumor Detected**\n\nConfidence: **{confidence:.2f}%**")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"✅ **No Tumor Detected**\n\nConfidence: **{confidence:.2f}%**")
