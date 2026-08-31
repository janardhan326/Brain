"""
Brain Tumor Detection from MRI Images
Capstone Project - Full Pipeline: Preprocessing -> Model -> Training -> Evaluation

Dataset expected structure (Kaggle "Brain MRI Images for Brain Tumor Detection"
or similar binary-labeled sets):
brain_tumor_dataset/
    yes/*.jpg   (tumor present)
    no/*.jpg    (no tumor)

This script auto-splits into train/val/test (no manual folder split needed).
Produces brain_tumor_model.keras — the exact filename app.py expects.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "brain_tumor_dataset"   # contains subfolders: yes/  no/
IMG_SIZE = (224, 224)              # MUST match app.py's cv2.resize target
BATCH_SIZE = 32
EPOCHS = 25
CLASS_NAMES = ["no", "yes"]        # index 0 = No Tumor, index 1 = Tumor
                                    # matches app.py: probability >= 0.5 -> "Tumor Detected"
MODEL_OUT = "brain_tumor_model.keras"

# ---------------------------------------------------------------------------
# 1. DATA PREPROCESSING (dataset has no pre-made train/val/test folders,
#    so we split it ourselves: 70% train, 15% val, 15% test)
# ---------------------------------------------------------------------------
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode="nearest",
    validation_split=0.30,
)
holdout_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.30)

train_gen = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=CLASS_NAMES,
    subset="training", shuffle=True, seed=42
)
holdout_gen = holdout_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=CLASS_NAMES,
    subset="validation", shuffle=False, seed=42
)

# Split the 30% holdout evenly into val / test
n_holdout = holdout_gen.samples
holdout_filepaths = holdout_gen.filepaths
holdout_labels = holdout_gen.classes
val_size = n_holdout // 2
val_files, val_labels = holdout_filepaths[:val_size], holdout_labels[:val_size]
test_files, test_labels = holdout_filepaths[val_size:], holdout_labels[val_size:]


def load_images(filepaths):
    imgs = []
    for fp in filepaths:
        img = tf.keras.preprocessing.image.load_img(fp, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        imgs.append(arr)
    return np.array(imgs)


X_val, y_val = load_images(val_files), np.array(val_labels)
X_test, y_test = load_images(test_files), np.array(test_labels)

print(f"Train: {train_gen.samples} | Val: {len(y_val)} | Test: {len(y_test)}")

# Handle class imbalance
labels = train_gen.classes
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# ---------------------------------------------------------------------------
# 2. MODEL - Transfer Learning with EfficientNetB0
# ---------------------------------------------------------------------------
def build_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False,
                                 input_tensor=Input(shape=(*IMG_SIZE, 3)))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model, base_model


model, base_model = build_model()
model.summary()

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUT, monitor="val_auc", mode="max", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
]

# ---------------------------------------------------------------------------
# 3. TRAINING - Phase 1: frozen base
# ---------------------------------------------------------------------------
history1 = model.fit(
    train_gen, validation_data=(X_val, y_val), epochs=EPOCHS,
    class_weight=class_weight_dict, callbacks=callbacks
)

# ---------------------------------------------------------------------------
# 3b. FINE-TUNING - Phase 2: unfreeze last layers
# ---------------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall")])

history2 = model.fit(
    train_gen, validation_data=(X_val, y_val), epochs=12,
    class_weight=class_weight_dict, callbacks=callbacks
)

model.save(MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# ---------------------------------------------------------------------------
# 4. EVALUATION (held-out test set — never seen during training/tuning)
# ---------------------------------------------------------------------------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["No Tumor", "Tumor"]))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Tumor", "Tumor"], yticklabels=["No Tumor", "Tumor"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history1.history["accuracy"] + history2.history["accuracy"], label="train")
plt.plot(history1.history["val_accuracy"] + history2.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history1.history["loss"] + history2.history["loss"], label="train")
plt.plot(history1.history["val_loss"] + history2.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")

print("\nSaved: confusion_matrix.png, roc_curve.png, training_curves.png")
