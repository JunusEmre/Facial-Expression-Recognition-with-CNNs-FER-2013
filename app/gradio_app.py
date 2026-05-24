from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_HEIGHT = 48
IMG_WIDTH = 48


# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# OpenCV face detector
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


# -----------------------------
# Face preprocessing
# -----------------------------
def preprocess_face(input_image):
    """
    Detects and crops the face from an uploaded image.
    If no face is detected, it uses the full image.
    Returns the model input and the processed 48x48 face image.
    """

    # Convert PIL image to RGB numpy array
    image_rgb = np.array(input_image.convert("RGB"))

    # Convert RGB to grayscale for face detection
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) > 0:
        # Select the largest detected face
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

        # Add a small margin around the face
        margin = int(0.20 * max(w, h))

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, gray.shape[1])
        y2 = min(y + h + margin, gray.shape[0])

        face = gray[y1:y2, x1:x2]
        face_detected = True
    else:
        # Fallback: use the full grayscale image
        face = gray
        face_detected = False

    # Resize to FER-2013 format
    face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize
    face_array = face_resized.astype("float32") / 255.0

    # Shape: (48, 48) -> (48, 48, 1)
    face_array = np.expand_dims(face_array, axis=-1)

    # Shape: (48, 48, 1) -> (1, 48, 48, 1)
    model_input = np.expand_dims(face_array, axis=0)

    # For display in Gradio
    processed_face_image = Image.fromarray(face_resized)

    return model_input, processed_face_image, face_detected


# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(input_image):
    """
    Takes an uploaded image, detects/crops the face,
    preprocesses it like FER-2013 data, and predicts emotion.
    """

    if input_image is None:
        empty_df = pd.DataFrame({
            "Emotion": ["No image uploaded"],
            "Confidence": ["0.00%"]
        })
        return {"No image": 1.0}, empty_df, None, "No image uploaded."

    model_input, processed_face_image, face_detected = preprocess_face(input_image)

    predictions = model.predict(model_input, verbose=0)[0]

    prediction_dict = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    probability_df = pd.DataFrame({
        "Emotion": CLASS_NAMES,
        "Confidence": predictions
    })

    probability_df = probability_df.sort_values(
        by="Confidence",
        ascending=False
    )

    top_emotion = probability_df.iloc[0]["Emotion"]
    top_confidence = float(probability_df.iloc[0]["Confidence"])

    probability_df["Confidence"] = probability_df["Confidence"].apply(
        lambda value: f"{value * 100:.2f}%"
    )

    if face_detected:
        status = (
            f"Face detected and cropped. "
            f"Top prediction: {top_emotion} ({top_confidence * 100:.2f}%)."
        )
    else:
        status = (
            f"No face was detected, so the full image was used. "
            f"Top prediction: {top_emotion} ({top_confidence * 100:.2f}%). "
            f"The result may be less reliable."
        )

    return prediction_dict, probability_df, processed_face_image, status


# -----------------------------
# Gradio interface
# -----------------------------
description = """
Upload or capture a face image and the CNN model will predict one of seven facial expressions:

angry, disgust, fear, happy, neutral, sad, or surprise.

The app first tries to detect and crop the face before sending it to the model.
"""

article = """
### Notes

This app is a portfolio demo for a deep learning project.

The model was trained on the FER-2013 dataset, which contains small 48x48 grayscale cropped face images.

Real-world camera images can be harder for the model because they may include background, different lighting, different camera angles and face positions.

This model should be used for learning and experimentation, not for important decisions about people.
"""

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(
        label="Upload or capture a face image",
        type="pil"
    ),
    outputs=[
        gr.Label(
            label="Predicted Emotion",
            num_top_classes=7
        ),
        gr.Dataframe(
            label="Prediction Confidence Table",
            headers=["Emotion", "Confidence"]
        ),
        gr.Image(
            label="Processed 48x48 Face Used by the Model",
            type="pil"
        ),
        gr.Textbox(
            label="Processing Status"
        )
    ],
    title="Facial Expression Recognition with CNN",
    description=description,
    article=article,
    theme="soft"
)


if __name__ == "__main__":
    demo.launch()