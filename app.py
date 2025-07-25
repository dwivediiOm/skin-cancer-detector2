import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('model.h5')

# Category mapping
category = {0: 'Benign', 1: 'Malignant'}

st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image to detect if it's **benign** or **malignant**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        img = img.resize((299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = model.predict(img_array)
        result = np.argmax(prediction)
        confidence = prediction[0][result]

        st.markdown(f"### Prediction: `{category[result]}`")
        st.markdown(f"### Confidence: `{confidence:.2f}`")
