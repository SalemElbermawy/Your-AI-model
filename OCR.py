
import shutil
import pytesseract

pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "/usr/bin/tesseract"




import streamlit as st
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("Extract ")
image_ocr=st.file_uploader("Upload :",type=["png", "jpg", "jpeg"])

if image_ocr is not None:
    image = image_ocr.read()
    np_imgage = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(np_imgage, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(img_rgb, output_type=Output.DICT, lang="ara+eng")

    for i, word in enumerate(data['text']):
        if word.strip() != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2) 

    st.image(img_rgb, caption="Detected Text", use_container_width=True)

    filtered_data = {
        "Text": [word for word in data['text'] if word.strip() != ""],
        "X": [data['left'][i] for i, word in enumerate(data['text']) if word.strip() != ""],
        "Y": [data['top'][i] for i, word in enumerate(data['text']) if word.strip() != ""],
        "Width": [data['width'][i] for i, word in enumerate(data['text']) if word.strip() != ""],
        "Height": [data['height'][i] for i, word in enumerate(data['text']) if word.strip() != ""]
    }

    df = pd.DataFrame(filtered_data)

    st.subheader("Extracted Text Data")
    st.dataframe(df)
    
    all_text = " ".join(df["Text"].tolist())
    st.write("text")
    st.text(all_text)

