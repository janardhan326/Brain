import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import os


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = "brain_tumor_model.keras"
IMG_SIZE = (224, 224)   # MUST match the size used in train.py


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        return model

    except Exception as e:
        st.error("Error loading the model:")
        st.code(str(e))
        return None


model = load_model()


# --------------------------------------------------
# CHECK MODEL
# --------------------------------------------------

if model is None:

    st.error(
        "❌ Brain tumor model was not loaded."
    )

    st.write(
        """
        Make sure that `brain_tumor_model.keras`
        is present in the same folder as `app.py`.
        Run `train.py` first to generate it.
        """
    )

    st.code(
        """
        brain_tumor/
        │
        ├── app.py
        ├── train.py
        ├── brain_tumor_model.keras
        └── requirements.txt
        """
    )

    st.stop()


# --------------------------------------------------
# FIX: validate the loaded model actually matches what
# this app expects, instead of assuming silently.
# --------------------------------------------------

expected_channels = 3
model_input_shape = model.input_shape  # e.g. (None, 224, 224, 3)

if len(model_input_shape) != 4 or model_input_shape[-1] != expected_channels:
    st.error(
        f"⚠️ Loaded model has unexpected input shape {model_input_shape}. "
        f"This app expects (None, H, W, 3)."
    )
    st.stop()

MODEL_IMG_SIZE = (model_input_shape[1], model_input_shape[2])


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Brain Tumor Detection System")

st.write(
    """
    Upload a brain MRI image and the trained
    CNN model will predict the image class.
    """
)

st.warning(
    """
    ⚠️ Educational/research prototype only.
    This application is not a medical diagnostic tool.
    Do not use the prediction for medical decisions.
    """
)


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

with st.expander("🔧 Model Information"):
    st.write("Model input shape:")
    st.write(model.input_shape)
    st.write("Model output shape:")
    st.write(model.output_shape)


# --------------------------------------------------
# UPLOAD IMAGE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # FIX: catch invalid/corrupt image uploads specifically,
    # instead of only catching errors after display.
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ That file isn't a valid image. Please upload a JPG or PNG.")
        st.stop()

    st.subheader("📷 Uploaded MRI Image")
    st.image(image, caption="Brain MRI", use_container_width=True)

    if st.button("🔍 Analyze MRI", use_container_width=True):

        with st.spinner("Analyzing MRI image..."):

            try:
                # FIX: use PIL for resizing (matches train.py's
                # tf.keras load_img/img_to_array pipeline exactly)
                # instead of cv2, which caused a preprocessing
                # mismatch between training and inference.
                img_resized = image.resize(MODEL_IMG_SIZE)
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array, verbose=0)
                probability = float(prediction[0][0])

            except Exception as e:
                st.error("❌ Error while running the model.")
                st.code(str(e))
                st.stop()

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.divider()
        st.subheader("🤖 Prediction Result")

        if probability >= 0.5:
            result = "Tumor Detected"
            confidence = probability * 100
            st.error(f"⚠️ {result}")
        else:
            result = "No Tumor Detected"
            confidence = (1 - probability) * 100
            st.success(f"✅ {result}")

        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.subheader("📊 Tumor Probability")
        st.progress(min(max(probability, 0.0), 1.0))
        st.write(f"Tumor probability: {probability * 100:.2f}%")

        # FIX: low-confidence predictions near the 0.5 boundary
        # are flagged instead of shown with false certainty.
        if 0.4 <= probability <= 0.6:
            st.info(
                "ℹ️ This prediction is close to the decision boundary — "
                "confidence is low. Treat this result with extra caution."
            )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("📚 About This Project")

st.write(
    """
    This project uses a Convolutional Neural Network
    (CNN, transfer learning on EfficientNetB0) trained on
    brain MRI images.

    The model performs binary image classification:

    • Class 0 → No Tumor
    • Class 1 → Tumor

    Image preprocessing (identical at train and inference time):

    • Resize image to match model input size
    • Normalize pixel values to [0, 1]
    • Add batch dimension
    • Send image to CNN model
    """
)

st.subheader("🛠️ Technologies Used")
st.write("Python | TensorFlow | Keras | NumPy | Pillow | Streamlit")
