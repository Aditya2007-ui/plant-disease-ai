import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Plant Disease AI", page_icon="🌿", layout="wide")

custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-color: #f4f9f4;
    }
    .title-text {
        color: #2e7d32;
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle-text {
        color: #555555;
        text-align: center;
        font-size: 20px;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .author-text {
        text-align: center;
        font-size: 16px;
        color: #888888;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<p class="title-text">🌿 AI Crop Doctor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Instantly detect and treat plant diseases using deep learning</p>', unsafe_allow_html=True)
st.markdown('<p class="author-text">Project by <b>Aditya Pagare</b></p>', unsafe_allow_html=True)
st.write("---")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_disease_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"System Frror:{e}")
    st.stop()

class_names = [
    "Apple - Scab", "Apple - Black Rot", "Apple - Cedar Rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight", "Corn - Healthy",
    "Grape - Black Rot", "Grape - Esca", "Grape - Leaf Blight", "Grape - Healthy",
    "Orange - Citrus Greening", "Peach - Bacterial Spot", "Peach - Healthy",
    "Pepper Bell - Bacterial Spot", "Pepper Bell - Healthy",
    "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy",
    "Raspberry - Healthy", "Soybean - Healthy", "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch", "Strawberry - Healthy",
    "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
    "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites",
    "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus", "Tomato - Healthy"
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Leaf Image")
    uploaded_file = st.file_uploader("Drag and drop a clear photo of the leaf", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

with col2:
    st.subheader("🔬 Diagnosis Results")
    
    if uploaded_file is None:
        st.info("Upload an image on the left to see the AI analysis here.")
    else:
        if st.button("Run AI Analysis", type="primary", use_container_width=True): 
            with st.spinner("Analyzing cell structure..."):
                try:
                    resized_image = image.resize((224, 224))
                    image_array = np.array(resized_image) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
                    
                    predictions = model.predict(image_array)
                    predicted_class_index = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    
                    detected_disease = class_names[predicted_class_index]
                    
                    st.success("Analysis Complete!")
                    st.metric(label="Detected Disease", value=detected_disease, delta=f"{confidence:.2f}% Confidence")
                    
                    if "Healthy" in detected_disease:
                        st.balloons()
                        st.success("*Status:* The plant is healthy! No treatment required.")
                    else:
                        st.warning("*Recommended Action:* Isolate the plant. Consult agricultural guidelines for disease-specific fungicides.")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
