# Brain Tumor Detection from MRI Images

Capstone project: CNN-based binary classification (Tumor / No Tumor)
from brain MRI images, transfer learning on EfficientNetB0, deployed
as a Streamlit app.

## Dataset
Kaggle "Brain MRI Images for Brain Tumor Detection" (or equivalent) —
binary folders `yes/` (tumor) and `no/` (no tumor).
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

## Project Structure
```
brain_tumor/
├── train.py              # preprocessing, model, training, evaluation
├── gradcam.py             # Grad-CAM interpretability
├── app.py                 # Streamlit deployment app (corrected)
├── requirements.txt
└── README.md
```

## What Was Fixed From the Original app.py

The app you shared assumed a model already existed but had no matching
training script, plus a few issues that would cause silent failures or
wrong predictions in practice:

1. **No training script existed.** `train.py` is new — it produces
   `brain_tumor_model.keras` with the exact filename, input size, and
   class order (`no`=0, `yes`=1) the app expects.
2. **Preprocessing mismatch (the most important fix).** The original
   app resized with `cv2.resize`, but a Keras model trained via
   `ImageDataGenerator`/`load_img` uses PIL's resizing internally —
   these use different interpolation and can shift predictions at
   inference time. `app.py` now resizes with PIL, matching `train.py`
   exactly, and reads the target size from the model itself
   (`model.input_shape`) instead of hardcoding `224x224` twice.
3. **No model/input validation.** If a mismatched or corrupted model
   file were loaded, the app would fail deep inside `.predict()` with
   a confusing error. It now validates the model's input shape at
   startup and fails with a clear message.
4. **No handling for invalid uploads.** A non-image file with a valid
   extension would crash inside the `try` block after already showing
   a broken image. Now caught explicitly with `UnidentifiedImageError`
   before display.
5. **No uncertainty signal.** A probability of 0.51 was shown with the
   same confident styling as 0.99. The app now flags predictions near
   the 0.5 decision boundary as low-confidence.
6. **No evaluation existed anywhere.** `train.py` adds class-weighted
   training (MRI tumor datasets are rarely perfectly balanced),
   precision/recall/F1, ROC-AUC, and a confusion matrix — accuracy
   alone is not a reliable enough metric for a medical classifier.

## How to Run

### 1. Train
```bash
pip install -r requirements.txt
python train.py
```
Place your dataset as `brain_tumor_dataset/yes/` and
`brain_tumor_dataset/no/` next to `train.py` first.
Produces `brain_tumor_model.keras`, `confusion_matrix.png`,
`roc_curve.png`, `training_curves.png`.

### 2. Explain a prediction
```bash
python gradcam.py path/to/mri.jpg
```

### 3. Deploy
```bash
streamlit run app.py
```
To share publicly for free: push this folder (with the trained
`.keras` file) to a Hugging Face Space with SDK = Streamlit, or to
Streamlit Community Cloud connected to a GitHub repo.

## Limitations & Ethical Considerations
- Small, single-source datasets of this type (~250 images) are prone
  to overfitting; results won't generalize to different scanners or
  populations without a larger, more diverse dataset.
- Binary tumor/no-tumor classification does not localize, size, or
  type the tumor — a real clinical tool would need segmentation and
  radiologist review.
- This is an educational prototype, not a validated diagnostic device.
  False negatives (missed tumors) are the highest-risk failure mode;
  the low-confidence flag in the app is a partial mitigation, not a
  substitute for clinical-grade validation.

## Results
(Fill in after running `train.py` — accuracy, AUC, precision/recall,
and the saved plots.)
