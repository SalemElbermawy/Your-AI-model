import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO


st.title("🖼️ Object Detection with YOLOv8")
st.markdown(
    """
    ### 🔍 **What does this model do?**
    This model uses **YOLOv8** (You Only Look Once) to detect objects in your uploaded image.  
    - ✅ **Upload an image** in PNG/JPG/JPEG format.  
    - ✅ The model will analyze it and draw **bounding boxes** around detected objects.  
    - ✅ YOLO is one of the fastest and most accurate object detection models.  

    ---
    **📌 Steps to use:**  
    1️⃣ Click **Browse files** and upload your image.  
    2️⃣ Wait for the model to process.  
    3️⃣ See the detected objects highlighted in your image!  
    """
)

image = st.file_uploader("📤 Upload your Image :", type=["png", "jpg", "jpeg"])

if image is not None:
    image = image.read()
    np_image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    model = YOLO('yolov8n.pt')

    results_image = model.predict(image, save=False, show=False)

    result_image = results_image[0].plot()
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    st.image(result_image, caption="✅ Objects Detected Successfully!", use_container_width=True)

    st.success("🎯 Detection completed! The objects in your image have been highlighted with bounding boxes.")
