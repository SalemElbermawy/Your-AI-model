import streamlit as st

st.set_page_config(
    page_title="Your AI Model",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Welcome to Your AI Models!")
st.header("✨ Do what you want with AI Power! 🚀")

st.sidebar.success("✅ Choose what you want")

st.markdown("""
<style>
.description {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 12px;
    font-size: 18px;
    color: #333333;
    line-height: 1.6;
}
.section-title {
    font-size: 22px;
    color: #4CAF50;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="description">
<p class="section-title">📌 Project Overview</p>
Welcome to our **AI-powered platform**! This project is divided into "3 main sections":

---

<p class="section-title">1️⃣ Model Training & Data Analysis</p>
- 📊 Upload your "custom dataset".  
- 🏆 Choose from "7+ powerful machine learning models".  
- 🔍 Explore your data with **interactive graphs** for both numerical and categorical features.  

---

<p class="section-title">2️⃣ Image Object Detection</p>
- 🖼 Upload any image.  
- 🎯 Detect and highlight all **objects inside the image** with advanced AI.  

---

<p class="section-title">3️⃣ OCR - Extract Text from Images</p>
- 🔡 Upload an image containing text.  
- ✅ Get all the text extracted **instantly** in a clean format.  

---

✨ Start exploring and enjoy the **power of AI**! 🚀
</div>
""", unsafe_allow_html=True)
