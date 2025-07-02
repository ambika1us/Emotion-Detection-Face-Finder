# 😃 Emotion Detection - Face Finder

A real-time facial emotion recognition app built with **TensorFlow**, **OpenCV**, and **Streamlit**. Instantly identify emotions like Happy, Angry, Sad, and more — either by uploading an image or capturing one from your webcam.

---

## 🚀 Live Demo
👉 [Try it out on Streamlit Cloud](https://emotion-detection-face-finder.streamlit.app/)

---

## 🧠 Features
- 🔍 **Face Detection** using Haar cascades (OpenCV)
- 🤖 **Emotion Classification** with a custom-trained deep learning model
- 🎯 **Real-time camera capture** or image upload
- 🎨 **Dynamic label styling** with emotion-specific color overlays
- 📈 Optional: Show top 3 emotion confidences or prediction bar chart

---

## 🛠 Tech Stack
- **Streamlit** – for the interactive UI
- **TensorFlow/Keras** – for loading and predicting with the model
- **OpenCV** – for face detection and drawing overlays
- **NumPy** & **Pillow** – image preprocessing and conversion

---

## 📦 Deploying to Streamlit Cloud
No fuss! Just push this folder to your GitHub repo and connect it to Streamlit Cloud. Streamlit will handle the environment setup and hosting automatically.
Make sure:
- runtime.txt contains python-3.10
- Your model and Haar cascade file are included
- Your requirements.txt reflects accurate versions

## 🤔 Example Model Prediction
Input: Image of a neutral face
Output: Neutral (93.3%)
Face highlighted with a gray label box.


## 🧠 Acknowledgments
- Pre-trained Haar cascades from OpenCV
- TensorFlow/Keras for model serialization and inference
- Streamlit for rapid prototyping
