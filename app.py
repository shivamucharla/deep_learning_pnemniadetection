import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('chest_model_v5.keras')  # Replace 'chest_model_v3.keras' with your actual model filename

# Preprocess the image correctly for the model
def preprocess_image(image):
    # Adjust the size to (256, 256) as required by the model
    image = image.resize((256, 256))  
    image = np.array(image)

    # Ensure the image has three channels (convert grayscale to RGB)
    if image.ndim == 2:  # If grayscale
        image = np.stack([image]*3, axis=-1)

    image = np.expand_dims(image, axis=0)
    return image / 255.0

# Streamlit app layout
st.title("Pneumonia Detection App")

# Upload an image
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess and make prediction
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)

    # Display the prediction (adjust based on your model's output)
    result = "Pneumonia Detected" if prediction[0][0] > 0.8 else "No Pneumonia Detected"
    st.write("Prediction:", result)
