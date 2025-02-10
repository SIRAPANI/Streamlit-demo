import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("model.h5")

# Define class labels (Modify based on your dataset)
class_labels = ["healthy", "diseased_leaf_spot","diseased_rust"]

# Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image to check its condition.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class index
    result = class_labels[predicted_class]

    # Display the result
    st.write(f"🌱 **Prediction:** {result}")
