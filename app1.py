import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load Haar cascade and model
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model('emotion_model.hdf5', compile=False)

# Emotion class labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
lang = st.selectbox("🌐 Select Language", ["English", "Hindi", "Odia"])

# Translations
emotion_translations = {
    'English': emotion_labels,
    'Hindi': ['गुस्सा', 'घृणा', 'डर', 'खुश', 'उदास', 'आश्चर्य', 'सामान्य'],
    'Odia': ['କ୍ରୋଧିତ', 'ଘୃଣା', 'ଭୟ', 'ସନ୍ତୁଷ୍ଟ', 'ଦୁଃଖିତ', 'ଆଶ୍ଚର୍ୟ', 'ସାଧାରଣ']
}

# Load Odia font for PIL
try:
    odia_font_path = "fonts/NotoSansOriya-Regular.ttf"
    hindi_font_path="fonts/NotoSansDevanagari-Regular.ttf"
    odia_font = ImageFont.truetype(odia_font_path, size=24)
    hindi_font = ImageFont.truetype(hindi_font_path, size=24)
except:
    odia_font = ImageFont.load_default()
    hindi_font = ImageFont.load_default()

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


# UI
st.title("😃 Emotion Detection – Face Finder")
#lang = st.selectbox("🌐 Select Language", ["English", "Hindi", "Odia"])
#uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
mode = st.radio("Select Input Mode", ["Upload", "Camera"])


if mode == "Upload":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Convert to RGB and wrap with PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi = roi_resized.astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            prediction = model.predict(roi)[0]
            max_index = int(np.argmax(prediction))
            confidence = prediction[max_index]

            base_emotion = emotion_labels[max_index]
            display_emotion = emotion_translations[lang][max_index]
            label_text = f"{display_emotion} ({confidence:.1%})"

            # Draw bounding box
            draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

            # Draw label using PIL (supports Unicode)
            # font_to_use = odia_font if lang == "Odia" else ImageFont.load_default()
            font_to_use = hindi_font if lang == "Hindi" else odia_font if lang == "Odia" else ImageFont.load_default()

            # text_width, text_height = draw.textsize(label_text, font=font_to_use)
            bbox = draw.textbbox((0, 0), label_text, font=font_to_use)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x + w // 2 - text_width // 2
            text_y = y - text_height - 10
            draw.text((text_x, text_y), label_text, font=font_to_use, fill=(255, 255, 255))

        final_img = np.array(pil_img)
        st.image(final_img, caption="🧠 Emotion(s) detected", use_container_width=True)

elif mode == "Camera":
    camera_image = st.camera_input("📷 Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image)
        image = np.array(image.convert('RGB'))
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Convert to RGB and wrap with PIL
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

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

                prediction = model.predict(roi)[0]
                max_index = int(np.argmax(prediction))
                confidence = prediction[max_index]

                # Color-coded label
                label_color = emotion_colors.get(confidence, (255, 255, 255))
                display_emotion = emotion_translations[lang][max_index]
                label = f"{display_emotion} ({confidence:.1%})"

                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

                font_to_use = hindi_font if lang == "Hindi" else odia_font if lang == "Odia" else ImageFont.load_default()

                bbox = draw.textbbox((0, 0), label, font=font_to_use)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x + w // 2 - text_width // 2
                text_y = y - text_height - 10
                draw.text((text_x, text_y), label, font=font_to_use, fill=(255, 255, 255))

            final_img = np.array(pil_img)
            st.image(final_img, caption="🧠 Emotion(s) detected", use_container_width=True)




