import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline
from pdf2image import convert_from_bytes

# Load summarizer (lightweight model for Hugging Face/Streamlit Cloud)
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp"

from transformers import pipeline
summarizer = pipeline("summarization", model="t5-small")


st.title("📄 Document Summary Assistant")
st.write("Upload a PDF or Image, and get a concise summary using AI.")

# File upload
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    text = ""

    # 📄 If it's a PDF
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        # 🔁 Fallback: if no text, try OCR on PDF pages
        if not text.strip():
            uploaded_file.seek(0)  # reset file pointer
            pdf_images = convert_from_bytes(uploaded_file.read())
            for img in pdf_images:
                text += pytesseract.image_to_string(img)

    # 🖼️ If it's an image
    else:
        img = Image.open(uploaded_file)
        text = pytesseract.image_to_string(img)

    # ✅ Summarization
    if text.strip():
        # Trim text if too long for model
        if len(text) > 1000:
            text = text[:1000]

        try:
            summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Summarization failed: {e}")
    else:
        st.warning("⚠️ No readable text found in this file. Try another document.")

