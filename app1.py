import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model

# --- Load assets ---
model = load_model('emotion_model.hdf5', compile=False)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_translations = {
    'English': emotion_labels,
    'Hindi': ['गुस्सा', 'घृणा', 'डर', 'खुश', 'उदास', 'आश्चर्य', 'सामान्य'],
    'Odia':   ['କ୍ରୋଧିତ', 'ଘୃଣା', 'ଭୟ', 'ସନ୍ତୁଷ୍ଟ', 'ଦୁଃଖିତ', 'ଆଶ୍ଚର୍ୟ', 'ସାଧାରଣ']
}

emotion_colors = {
    'Angry': (0, 0, 255), 'Disgust': (0, 128, 0), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 255), 'Sad': (255, 0, 0), 'Surprise': (255, 165, 0),
    'Neutral': (128, 128, 128)
}

try:
    odia_font = ImageFont.truetype("fonts/NotoSansOriya-Regular.ttf", size=32)
    hindi_font = ImageFont.truetype("fonts/NotoSansDevanagari-Regular.ttf", size=32)
except:
    odia_font = hindi_font = ImageFont.load_default("fonts/NotoSans-Regular.ttf", size=32)


# --- Helper functions ---
def prepare_face_roi(gray_img, x, y, w, h):
    roi_gray = gray_img[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi_gray, (48, 48))
    roi = roi_resized.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)[..., np.newaxis]
    return roi

def select_language_font(lang):
    if lang == "Hindi": return hindi_font
    if lang == "Odia":  return odia_font
    return ImageFont.load_default()

def draw_emotion_label(draw, box, label_text, font):
    x, y, w, h = box
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx, ty = x + w//2 - tw//2, y - th - 10
    draw.text((tx, ty), label_text, font=font, fill=(255, 255, 255))


# --- UI ---
st.title("😃 Emotion Detection – Face Finder")
lang = st.selectbox("🌐 Select Language", ["English", "Hindi", "Odia"])
mode = st.radio("📤 Select Input Mode", ["Upload", "Camera"])

img_data = None
if mode == "Upload":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_data = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
else:
    photo = st.camera_input("Take a photo")
    if photo:
        img_data = np.array(Image.open(photo).convert("RGB"))[:, :, ::-1]  # RGB to BGR

# --- Process image if available ---
if img_data is not None:
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    pil_img = Image.fromarray(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for (x, y, w, h) in faces:
        roi = prepare_face_roi(gray, x, y, w, h)
        prediction = model.predict(roi)[0]
        max_idx = int(np.argmax(prediction))
        confidence = prediction[max_idx]

        base_emotion = emotion_labels[max_idx]
        display_emotion = emotion_translations[lang][max_idx]
        label_text = f"{display_emotion} ({confidence:.1%})"
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

        font = select_language_font(lang)
        draw_emotion_label(draw, (x, y, w, h), label_text, font)

    st.image(np.array(pil_img), caption="🧠 Emotion(s) Detected", use_container_width=True)