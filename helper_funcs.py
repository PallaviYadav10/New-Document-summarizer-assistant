import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import requests
import re

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    pages = []
    for p in doc:
        pages.append(p.get_text("text"))
    return "\n\n".join(pages)

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang='eng')

def chunk_text(text, max_chars=3000):
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            next_dot = text.rfind('.', start, end)
            if next_dot > start:
                end = next_dot + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def summarize_hf(text, hf_token, max_new_tokens=150):
    url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"inputs": text, "parameters": {"max_new_tokens": max_new_tokens}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and "summary_text" in data[0]:
        return data[0]["summary_text"]
    return str(data)

def summarize_large_text(text, hf_token, mode="short"):
    max_by_mode = {"short": 60, "medium": 150, "long": 300}
    tgt = max_by_mode.get(mode, 150)
    chunks = chunk_text(text, max_chars=3000)
    partials = [summarize_hf(c, hf_token, max_new_tokens=tgt) for c in chunks]
    if len(partials) > 1:
        joined = "\n".join(partials)
        return summarize_hf(joined, hf_token, max_new_tokens=tgt)
    return partials[0]

def extract_highlights(source_text, summary_text, top_n=3):
    sents = re.split(r'(?<=[.?!])\s+', source_text.strip())
    summary_words = set(re.findall(r'\w+', summary_text.lower()))
    scored = []
    for s in sents:
        words = re.findall(r'\w+', s.lower())
        score = sum(1 for w in words if w in summary_words)
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [s for (score,s) in scored[:top_n] if score>0]
