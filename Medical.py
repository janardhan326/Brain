import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
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
        """
    )

    st.code(
        """
        brain-tumor/
        │
        ├── app.py
        ├── brain_tumor_model.keras
        └── requirements.txt
        """
    )

    st.stop()


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

    st.write(
        "Model input shape:"
    )

    st.write(
        model.input_shape
    )

    st.write(
        "Model output shape:"
    )

    st.write(
        model.output_shape
    )


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

    try:

        # Open image
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        # Display image
        st.subheader(
            "📷 Uploaded MRI Image"
        )

        st.image(
            image,
            caption="Brain MRI",
            use_container_width=True
        )


        # --------------------------------------------------
        # ANALYZE BUTTON
        # --------------------------------------------------

        if st.button(
            "🔍 Analyze MRI",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing MRI image..."
            ):

                # Convert PIL → NumPy
                img = np.array(image)

                # Resize
                img = cv2.resize(
                    img,
                    (224, 224)
                )

                # Convert to float
                img = img.astype(
                    np.float32
                )

                # Normalize
                img = img / 255.0

                # Add batch dimension
                img = np.expand_dims(
                    img,
                    axis=0
                )

                # Prediction
                prediction = model.predict(
                    img,
                    verbose=0
                )

                # Get value
                probability = float(
                    prediction[0][0]
                )


            # --------------------------------------------------
            # RESULT
            # --------------------------------------------------

            st.divider()

            st.subheader(
                "🤖 Prediction Result"
            )


            if probability >= 0.5:

                result = "Tumor Detected"

                confidence = probability * 100

                st.error(
                    f"⚠️ {result}"
                )

            else:

                result = "No Tumor Detected"

                confidence = (
                    1 - probability
                ) * 100

                st.success(
                    f"✅ {result}"
                )


            # --------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


            # --------------------------------------------------
            # PROBABILITY
            # --------------------------------------------------

            st.subheader(
                "📊 Tumor Probability"
            )

            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )

            st.write(
                f"Tumor probability: "
                f"{probability * 100:.2f}%"
            )


    except Exception as e:

        st.error(
            "❌ Error while processing the image."
        )

        st.code(
            str(e)
        )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader(
    "📚 About This Project"
)

st.write(
    """
    This project uses a Convolutional Neural Network
    (CNN) trained on brain MRI images.

    The model performs binary image classification:

    • Class 0 → No Tumor
    • Class 1 → Tumor

    Image preprocessing:

    • Resize image to 224 × 224
    • Normalize pixel values
    • Add batch dimension
    • Send image to CNN model
    """
)


st.subheader(
    "🛠️ Technologies Used"
)

st.write(
    """
    Python | TensorFlow | Keras | OpenCV |
    NumPy | Pillow | Streamlit
    """
)
