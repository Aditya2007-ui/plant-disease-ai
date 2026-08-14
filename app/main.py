import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

CLASS_NAMES = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']

PLANT_INFO = {
    "Tomato - Early blight": "Early blight is a fungal disease that causes brown spots with concentric rings on lower leaves. *Treatment:* Remove affected leaves, ensure good air circulation, and apply a copper-based fungicide.",
    "Tomato - Late blight": "Late blight is a highly destructive disease that causes irregular, dark, water-soaked spots on leaves. *Treatment:* Remove and destroy all infected plant parts immediately. Apply a copper fungicide to protect healthy tissue.",
    "Tomato - healthy": "This tomato leaf looks perfectly healthy! Keep up the good work with proper watering and sunlight.",
    "Strawberry - Leaf scorch": "Leaf scorch is a fungal disease causing irregular purple or brown spots on the upper leaf surface. *Treatment:* Remove infected leaves and improve air circulation.",
    "Strawberry - healthy": "This strawberry leaf is healthy and shows no signs of fungal or bacterial disease!",
    "Apple - Apple scab": "Apple scab appears as olive-green to black spots on leaves and fruit. *Treatment:* Rake up fallen leaves in autumn and use fungicides preventatively in early spring.",
    "Corn(maize) - healthy": "Your corn is looking great and shows no signs of common leaf diseases!",
    "Tomato - Leaf Mold": "Leaf mold is a fungal disease that typically appears as pale green or yellow spots on the upper side of older leaves, with a velvety olive-green fungus on the underside. *Treatment:* Improve air circulation by pruning, avoid overhead watering, and use preventative copper fungicides."
}

st.set_page_config(page_title="Plant Disease Predictor", page_icon="🌿")
st.title("🌿 AI Plant Disease Identifier")
st.caption("Project by Aditya Pagare")
st.write("Upload a picture of a plant leaf, and the AI will diagnose it instantly!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/plant_disease_model.keras")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Uploaded Leaf', use_container_width=True)
    
    st.write("Analyzing...")
    
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    clean_name = predicted_class.replace("_", " - ").replace("_", " ").strip()
    
    st.success(f"*Prediction:* {clean_name}")
    st.info(f"*Confidence:* {confidence:.2f}%")
    
    st.divider() 
    st.subheader("📖 About this Prediction")
    
    extra_details = PLANT_INFO.get(clean_name, "Detailed information and treatment steps for this specific plant/disease will be added to our database soon!")
    st.write(extra_details)
