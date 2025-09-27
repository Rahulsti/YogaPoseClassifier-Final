# =======================================================
# YOGA POSE CLASSIFIER - STREAMLIT APP (`app.py`)
# =======================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import time
import io

# --- Configuration ---
IMG_SIZE = (224, 224)
MODEL_PATH = 'yoga_pose_model.h5' 

# !!! CRUCIAL: UPDATE THIS LIST WITH YOUR ACTUAL POSE NAMES !!!
# Example: If your dataset folders were 'Tadasana', 'Vriksasana', etc.
CLASS_LABELS = [
    'downdog',  # REPLACE with your first pose name
    'goddess',
    'plank',
    'tree',
    'warrior2',  # REPLACE with your second pose name
    # ... Add ALL your pose names here, matching the order they were processed in training
]

# --- Model Loading (Caching for efficiency) ---
@st.cache_resource 
def load_trained_model():
    """Loads the trained Keras model."""
    try:
        model = load_model(MODEL_PATH) 
        return model
    except Exception as e:
        st.error(f"Error loading model '{MODEL_PATH}'. Ensure the file is present and correct.")
        st.stop() 

# Load the model globally
model = load_trained_model()

# --- Prediction Logic ---
def preprocess_and_predict(img):
    """Preprocesses a PIL image and returns the prediction."""
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    preds = model.predict(img_array)
    
    class_idx = np.argmax(preds)
    pose_label = CLASS_LABELS[class_idx]
    confidence = float(preds[0][class_idx]) 

    return pose_label, confidence, preds[0]

# =======================================================
# --- Streamlit Frontend Layout ---
# =======================================================

st.title("🧘 Yoga Pose Classifier")
st.markdown("Upload an image to identify the yoga pose using your MobileNetV2 model.")

uploaded_file = st.file_uploader("Choose a yoga pose image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("") 

    if st.button('Classify Pose'):
        with st.spinner('Analyzing the pose...'):
            time.sleep(1) # Visual effect
            
            pose_label, confidence, raw_predictions = preprocess_and_predict(img)
            
            st.success(f"Prediction Complete!")

            if confidence <0.9:
                st.warning("Not a yoga pose I recognize. Please try another image.")
            # --- Display Results ---
            else:
                st.subheader(f"Predicted Pose: **{pose_label}**")
                st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")

                # Show all probabilities in a chart
                st.subheader("Probability Distribution")
                
                df_results = pd.DataFrame({'Pose': CLASS_LABELS, 'Probability': raw_predictions})
                df_results = df_results.sort_values(by='Probability', ascending=False)
                
                st.bar_chart(df_results, x='Pose', y='Probability', color="#ff9900")