import os
import streamlit as st
from helper_funcs import extract_text_from_pdf, extract_text_from_image, summarize_large_text, extract_highlights
from pathlib import Path

st.set_page_config(page_title="Document Summary Assistant", layout="centered")
st.title("Document Summary Assistant (Python - Streamlit)")

# Token: prefer st.secrets (HuggingFace Spaces) or environment variable
HF_TOKEN = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else os.environ.get("HF_TOKEN", "")

mode = st.radio("Summary length", ["short", "medium", "long"])
uploaded = st.file_uploader("Upload PDF or image", type=["pdf","png","jpg","jpeg","tiff"])

if not HF_TOKEN:
    st.warning("Please add your Hugging Face API token in Secrets (HF_TOKEN).")

if uploaded:
    tmpdir = Path("/tmp/uploads")
    tmpdir.mkdir(parents=True, exist_ok=True)
    fpath = tmpdir / uploaded.name
    fpath.write_bytes(uploaded.getvalue())

    with st.spinner("Extracting text..."):
        try:
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                text = extract_text_from_pdf(str(fpath))
            else:
                text = extract_text_from_image(str(fpath))
        except Exception as e:
            st.error("Extraction error: " + str(e))
            st.stop()

    st.subheader("Extracted text (preview)")
    st.text_area("", text[:6000], height=220)

    if st.button("Generate summary"):
        if not HF_TOKEN:
            st.error("HF token missing. Add HF_TOKEN to Secrets and retry.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = summarize_large_text(text, HF_TOKEN, mode=mode)
                    highlights = extract_highlights(text, summary)
                except Exception as e:
                    st.error("Summarization error: " + str(e))
                    st.stop()

            st.subheader("Summary")
            st.write(summary)
            st.subheader("Key Highlights")
            for h in highlights:
                st.markdown(f"- {h}")
