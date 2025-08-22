import streamlit as st
import cv2
import numpy as np
image=st.file_uploader("Upload your Image :",type=["png", "jpg", "jpeg"])


from ultralytics import YOLO

if image is not None:


    image = image.read()
    np_imgage = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(np_imgage, cv2.IMREAD_COLOR)

    model = YOLO('yolov8n.pt')




    results_image = model.predict(image, save=False, show=False)

    result_image = results_image[0].plot()


    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)


    st.image(result_image, caption="OD", use_container_width=True)