import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

reader = easyocr.Reader(['ar', 'en'])

st.title("Extract Text from Images")
image_ocr = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])

if image_ocr is not None:
     
    image = image_ocr.read()
    np_image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    results = reader.readtext(img_rgb)
    
   
    for (bbox, text, confidence) in results:
        if text.strip() != "":
            
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2)
           
            cv2.putText(img_rgb, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    st.image(img_rgb, caption="Detected Text", use_container_width=True)

    
    filtered_data = {
        "Text": [text for (bbox, text, confidence) in results if text.strip() != ""],
        "Confidence": [confidence for (bbox, text, confidence) in results if text.strip() != ""],
        "X": [int(bbox[0][0]) for (bbox, text, confidence) in results if text.strip() != ""],
        "Y": [int(bbox[0][1]) for (bbox, text, confidence) in results if text.strip() != ""],
        "Width": [int(bbox[2][0] - bbox[0][0]) for (bbox, text, confidence) in results if text.strip() != ""],
        "Height": [int(bbox[2][1] - bbox[0][1]) for (bbox, text, confidence) in results if text.strip() != ""]
    }

    df = pd.DataFrame(filtered_data)

    st.subheader("Extracted Text Data")
    st.dataframe(df)
    
    all_text = " ".join(df["Text"].tolist())
    st.write("Full Extracted Text:")
    st.text(all_text)



# git add .
# git commit -m "refactor: completely rebuild pages folder with new OCR implementation"
# git push origin main