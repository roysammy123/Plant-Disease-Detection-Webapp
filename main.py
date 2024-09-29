import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.title('🌿 Plant Disease Detector')

st.markdown("""
Welcome to the Plant Disease Detector! This application uses advanced machine learning 
techniques to identify diseases in plants based on images of their leaves.

### How to use:
1. Upload a clear image of a plant leaf
2. Click the 'Classify' button
3. Get instant results about potential diseases

This tool can help farmers, gardeners, and plant enthusiasts quickly identify plant health issues.
""")

# About section moved to main page
st.markdown("### About")
st.info(
    "This app uses a deep learning model trained on thousands of plant images "
    "to detect various plant diseases. It can recognize multiple plant species "
    "and their associated diseases."
)

uploaded_image = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.write("### Image Analysis")
        if st.button('Classify', key='classify_button'):
            with st.spinner("Analyzing the image..."):
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
            st.balloons()

st.markdown("---")
st.markdown("### Disclaimer")
st.info(
    "This tool is for educational purposes only. For accurate diagnosis and treatment "
    "of plant diseases, please consult with a professional agriculturist or plant pathologist."
)

# Add GitHub buttons for team members
st.markdown("---")
st.markdown("### Our Team")

col1, col2, col3 = st.columns(3)

def github_button(name, github_url):
    return f'''
    <a href="{github_url}" target="_blank">
        <button style="width:90%; height:50px; font-size:16px; background-color:#1E90FF; color:white; border:none; border-radius:5px; cursor:pointer; margin:5px 0;">
            {name}
        </button>
    </a>
    '''

with col1:
    st.markdown(github_button("Soumyajit Roy", "https://github.com/roysammy123"), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)  # Spacer
    st.markdown(github_button("Ishtaj Kaur Deol", "https://github.com/ishtaj"), unsafe_allow_html=True)

with col2:
    st.markdown(github_button("Manav Malhotra", "https://github.com/Manav173"), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)  # Spacer
    st.markdown(github_button("Saksham Chhawan", "https://github.com/sakshamchhawan18"), unsafe_allow_html=True)

with col3:
    st.markdown(github_button("Swarnav Kumar", "https://github.com/Swarnav-Kumar"), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)  # Spacer
    st.markdown(github_button("Madhurima Aich", "https://github.com/Madhurima1826"), unsafe_allow_html=True)
