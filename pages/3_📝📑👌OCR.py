import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd

st.title("📜 OCR Text Extraction Tool")
st.markdown(
    """
    ### 🔍 Extract Text from Images  
    This feature allows you to:  
    - ✅ Upload an image in **PNG/JPG/JPEG** format  
    - ✅ Detect and extract **Arabic & English text**  
    - ✅ Display detected text with **bounding boxes**  
    - ✅ Get the text data in a structured **DataFrame**  

    ---
    **📌 Steps to use:**  
    1️⃣ Upload your image  
    2️⃣ Wait for processing  
    3️⃣ View highlighted text and copy extracted content  
    """
)


@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

image_ocr = st.file_uploader("📤 Upload Image:", type=["png", "jpg", "jpeg"])

if image_ocr is not None:
    
    image_bytes = image_ocr.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    with st.spinner('🔍 Processing image and detecting text...'):
        results = reader.readtext(img_rgb)
    
   
    for (bbox, text, confidence) in results:
        if text.strip() != "":
            
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2)
            
            
    
    st.image(img_rgb, caption="✅ Detected Text in Image", use_container_width=True)
    
  
    filtered_data = {
        "Text": [text for (_, text, _) in results if text.strip() != ""],
        "Confidence": [f"{confidence:.2%}" for (_, _, confidence) in results if text.strip() != ""],
        "Position": [f"{bbox[0]} to {bbox[2]}" for (bbox, _, _) in results if text.strip() != ""]
    }
    
    df = pd.DataFrame(filtered_data)
    st.subheader("📋 Extracted Text Data")
    st.dataframe(df)
    
    
    all_text = " ".join(df["Text"].tolist())
    st.markdown("### 📝 Full Extracted Text")
    st.text_area("Extracted Text", all_text, height=150)