import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load emotion recognition model
emotion_model = load_model('emotion_model.hdf5', compile=False)

# Emotion class labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion-to-color mapping (BGR format for OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 128, 0),     # Green
    'Fear': (128, 0, 128),      # Purple
    'Happy': (0, 255, 255),     # Yellow
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (255, 165, 0),  # Orange-ish
    'Neutral': (128, 128, 128)  # Gray
}

# Load Haar cascade once at the top
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("😃 Emotion Detection - Face Finder (Upload Mode)")

mode = st.radio("Select Input Mode", ["Upload", "Camera"])
if mode == "Upload":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces (using pre-trained Haar cascade or other method)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = grayscale[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi = roi_resized.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=0)[..., np.newaxis]

            # Get emotion prediction
            prediction = emotion_model.predict(roi)
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]
            confidence = float(np.max(prediction))

            # Draw rectangle and label
            # Get color for current emotion
            label_color = emotion_colors.get(predicted_emotion, (255, 255, 255))  # default to white

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{predicted_emotion} ({confidence:.1%})"
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.9, (255, 0, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, label_color, 2)

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

elif mode == "Camera":
    camera_image = st.camera_input("📷 Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image)
        image = np.array(image.convert('RGB'))
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("Failed to load Haar cascade.")
        else:
            faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = grayscale[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi = roi_resized.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)[..., np.newaxis]

                prediction = emotion_model.predict(roi)
                max_index = int(np.argmax(prediction))
                predicted_emotion = emotion_labels[max_index]
                confidence = float(np.max(prediction))

                # Color-coded label
                label_color = emotion_colors.get(predicted_emotion, (255, 255, 255))
                label = f"{predicted_emotion} ({confidence:.1%})"
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)

            st.image(image, channels="RGB")




