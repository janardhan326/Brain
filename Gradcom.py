"""
Grad-CAM visualization for the brain tumor model — shows which regions
of the MRI drove the prediction. Adds interpretability for the report.
"""

import sys
import numpy as np
import tensorflow as tf
import cv2

MODEL_PATH = "brain_tumor_model.keras"
IMG_SIZE = (224, 224)
# EfficientNetB0's last conv layer name (from train.py's base model)
LAST_CONV_LAYER = "top_conv"


def get_gradcam(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, alpha=0.4, out_path="gradcam_output.png"):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(out_path, superimposed)
    return superimposed


if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    heatmap = get_gradcam(model, arr, LAST_CONV_LAYER)
    overlay_heatmap(img_path, heatmap)
    print("Saved gradcam_output.png")
