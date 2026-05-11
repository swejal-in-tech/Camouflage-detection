import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = os.path.join(os.path.dirname(__file__), "unet_camouflage.h5")
model = tf.keras.models.load_model(model_path, compile=False)

st.title("Camouflage Animal Detection System")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = np.array(image.resize((256, 256)))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    mask = prediction[:, :, 0]

    st.image(mask)

    if np.mean(mask > 0.2) > 0.01:
        st.success("Animal FOUND!")
    else:
        st.error("No Animal Detected")
