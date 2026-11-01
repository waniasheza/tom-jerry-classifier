import numpy as np
import cv2
import joblib
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Tom & Jerry Classifier", page_icon="🐱")
st.title("🐱🐭 Tom & Jerry Image Classifier")
st.write("Upload a frame from the cartoon and the model will tell who is in it.")
st.caption("Features: ORB keypoints + Bag of Visual Words + ML classifier")

# Load the saved model
saved = joblib.load("tom_jerry_orb_model.pkl")
kmeans, model, classes, K = saved["kmeans"], saved["model"], saved["classes"], saved["K"]
orb = cv2.ORB_create(nfeatures=300)

names = {
    "tom": "Only Tom 🐱",
    "jerry": "Only Jerry 🐭",
    "tom_jerry_1": "Both Tom and Jerry 🐱🐭",
    "tom_jerry_0": "Neither Tom nor Jerry",
}

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    pil_img = Image.open(file).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    # Convert RGB to BGR because OpenCV works with BGR
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (256, 256))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Extract ORB features
    kp, des = orb.detectAndCompute(gray, None)

    if des is None:
        st.warning("No keypoints found in this image. Try another one.")
    else:
        # Build the visual-words histogram
        words = kmeans.predict(des.astype(np.float32))
        hist = np.bincount(words, minlength=K).astype(float)
        hist = hist / hist.sum()

        pred = model.predict([hist])[0]
        st.success(f"Prediction: {names.get(classes[pred], classes[pred])}")
        st.write(f"Keypoints found: {len(kp)}")
