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

CLASS_NAMES = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn(maize)_Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 'Grapehealthy', 'Orange_Haunglongbing(Citrus_greening)', 'PeachBacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

PLANT_INFO = {
    "early blight": "Early blight is a fungal disease that causes brown spots with concentric rings on lower leaves. Treatment: Remove affected leaves, ensure good air circulation, and apply a copper-based fungicide.",
    "late blight": "Late blight is a highly destructive disease that causes irregular, dark, water-soaked spots on leaves. Treatment: Remove and destroy all infected plant parts immediately. Apply a copper fungicide to protect healthy tissue.",
    "leaf scorch": "Leaf scorch is a fungal disease causing irregular purple or brown spots on the upper leaf surface. Treatment: Remove infected leaves and improve air circulation.",
    "apple scab": "Apple scab appears as olive-green to black spots on leaves and fruit. Treatment: Rake up fallen leaves in autumn and use fungicides preventatively in early spring.",
    "mold": "Leaf mold is a fungal disease that typically appears as pale green or yellow spots on the upper side of older leaves, with a velvety olive-green fungus on the underside. Treatment: Improve air circulation by pruning, avoid overhead watering, and use preventative copper fungicides.",
    "septoria": "Septoria leaf spot is a destructive fungal disease that causes numerous small, circular spots with dark borders and light gray centers. Treatment: Remove infected leaves immediately and apply a copper-based fungicide.",
    "cercospora": "Cercospora leaf spot (Gray leaf spot) is a fungal disease that creates rectangular, pale brown to gray spots on leaves. Treatment: Use crop rotation, remove plant debris, and apply appropriate fungicides if severe.",
    "healthy": "This leaf looks perfectly healthy! Keep up the good work with proper watering and sunlight."
}

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
    st.error(f"System Error: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Leaf Image")
    uploaded_file = st.file_uploader("Drag and drop a clear photo of the leaf", type=["jpg", "jpeg", "png", "webp"])
    
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
                    img = image.resize((224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    
                    predictions = model.predict(img_array, verbose=0)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = CLASS_NAMES[np.argmax(score)]
                    confidence = float(100 * np.max(score))
                    
                    if confidence < 5:
                        st.error("⚠️ Error: This does not look like a valid plant leaf!")
                        st.warning(f"The AI is only {confidence:.2f}% confident. Please upload a clear picture of a single leaf on a solid background.")
                    else:
                        clean_name = predicted_class.replace("_", " - ").replace("", " ").strip()
                        
                        st.success("Analysis Complete!")
                        st.metric(label="Detected Condition", value=clean_name, delta=f"{confidence:.2f}% Confidence", delta_color="off")
                        
                        extra_details = "Detailed information and treatment steps for this specific plant/disease will be added to our database soon!"
                        clean_name_lower = clean_name.lower()
                        
                        for keyword, info in PLANT_INFO.items():
                            if keyword in clean_name_lower:
                                extra_details = info
                                break
                        
                        if "healthy" in clean_name_lower:
                            st.balloons()
                            st.success(f"*Status:* {extra_details}")
                        else:
                            st.warning(f"*Treatment / Info:* {extra_details}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
