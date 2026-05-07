import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Custom CSS for Strict Single-Page UI
# -------------------------------
st.markdown("""
    <style>
    /* Prevent overall page scroll */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 3rem;
        padding-right: 3rem;
        height: 100vh;
        overflow: hidden;
    }

    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020c1b);
        color: #e6f1ff;
        overflow: hidden;
    }

    /* Tighten Vertical Spacing */
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }

    /* Glassmorphism Containers with reduced padding */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 8px;
    }

    /* Title Styling - More Compact */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #64ffda, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0px;
        font-size: 2rem !important;
    }

    /* Subtitle Styling */
    .subtitle {
        color: #8892b0;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }

    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #64ffda 0%, #48cae4 100%) !important;
        color: #0a192f !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.45rem !important;
        transition: all 0.3s ease !important;
    }

    /* Image Container - Strict Height and Width */
    .img-container {
        max-height: 200px;
        max-width: 250px;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 10px;
        background: rgba(0,0,0,0.2);
        margin: 0 auto;
    }
    
    .stImage > img {
        max-height: 180px !important;
        width: auto !important;
        object-fit: contain;
    }





    /* Result Card - Compact */
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        text-align: center;
    }

    /* Hide Streamlit Header/Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Adjust Info/Success/Warning boxes to be smaller */
    .stAlert {
        padding: 0.5rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model (with caching)
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("brain_tumor_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# UI Layout
# -------------------------------

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.title("NeuroScan AI")
    st.markdown('<p class="subtitle">Next-Gen Brain Tumor Detection</p>', unsafe_allow_html=True)
    
    st.write("Deep Neural Network for rapid MRI analysis.")
    
    st.info("💡 **Tip:** Use T1/T2 weighted high-res scans.")
    
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    
    predict_btn = st.button("🚀 Run AI Diagnosis")

    
    if uploaded_file is not None and predict_btn:
        if model is None:
            st.error("Model Error")
        else:
            with st.spinner("Analyzing scan..."):
                # Preprocess
                img = image.resize((224, 224))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)

                # Prediction
                prediction = model.predict(img)[0][0]

                # Display Result
                if prediction > 0.5:
                    confidence = prediction * 100
                    st.markdown(f"""
                        <div class="result-card" style="background: rgba(255, 75, 75, 0.1); border: 1px solid #ff4b4b;">
                            <h3 style="color: #ff4b4b; margin:0;">🚨 Tumor Detected</h3>
                            <p style="font-size: 1rem; margin: 5px 0; color: #e6f1ff;">Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning("Consult a specialist immediately.")
                else:
                    confidence = (1 - prediction) * 100
                    st.markdown(f"""
                        <div class="result-card" style="background: rgba(100, 255, 218, 0.1); border: 1px solid #64ffda;">
                            <h3 style="color: #64ffda; margin:0;">✅ No Tumor Detected</h3>
                            <p style="font-size: 1rem; margin: 5px 0; color: #e6f1ff;">Confidence: <b>{confidence:.2f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("Scan appears clear.")
    elif uploaded_file is None:
        st.markdown("""
            <div style="margin-top: 20px; padding: 15px; background: rgba(100, 255, 218, 0.05); border-radius: 8px; border-left: 4px solid #64ffda;">
                <p style="margin:0; font-size: 0.9rem; color: #64ffda;"><b>Waiting for Input...</b><br>Please upload a scan to begin clinical analysis.</p>
            </div>
        """, unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        # Display image with constrained height
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, width=550)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="height: 250px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.02); border-radius: 15px; border: 1px dashed rgba(255,255,255,0.1);">
                <div style="text-align: center; color: #8892b0;">
                    <h2 style="margin-bottom: 5px; font-size: 1.2rem;">Diagnostic Viewport</h2>
                    <p style="font-size: 0.8rem;">Scan results will appear here</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
    <div style="position: fixed; bottom: 10px; left: 3rem; color: rgba(136, 146, 176, 0.5); font-size: 0.75rem;">
        NeuroScan AI v2.0 | Advanced Clinical Intelligence | System Active
    </div>
""", unsafe_allow_html=True)



