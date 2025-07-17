import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the pre-trained model
model = tf.keras.models.load_model("mnist_nn_model-2.keras")

# Title
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a digit (0–9), and I'll predict what it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors: black digit on white bg

    # Resize to 28x28
    image = image.resize((28, 28))

    # Display the image
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess: Convert to numpy array
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28 * 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"🧾 I think this digit is: **{predicted_class}**")
