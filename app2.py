import streamlit as st
import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import spacy
import re

# Load models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
nlp = spacy.load("en_core_web_sm")

# --- Helper Functions ---

def clean_filename(filename, title_case=True):
    name_part = filename.rsplit(".", 1)[0]
    name_part = re.sub(r"\([^)]*\)", "", name_part)  # remove (35)
    name_part = name_part.replace("-", " ").replace("_", " ")
    name_part = " ".join(name_part.split())
    if title_case:
        name_part = name_part.title()
    return name_part

def generate_caption(image, filename, title_case=True):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    raw_caption = processor.decode(out[0], skip_special_tokens=True)

    # Clean filename
    name_part = clean_filename(filename, title_case)

    # Combine filename with caption
    final_caption = f"{name_part} {raw_caption}"
    return final_caption

def adjust_meta_length(text, min_len=150, max_len=160):
    """Ensure meta description length between 150–160 chars."""
    if len(text) > max_len:
        return text[:max_len].rsplit(" ", 1)[0] + "..."
    elif len(text) < min_len:
        padding = " Designed with premium craftsmanship, cutting-edge materials, and luxury sports car performance."
        return (text + padding)[:max_len]
    return text

def generate_keywords(caption, filename):
    doc = nlp(caption)

    # Extract meaningful multi-word noun phrases
    noun_chunks = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

    # Clean and remove duplicates
    keywords = list(dict.fromkeys(noun_chunks))

    # Add model/year info from filename
    base_name = clean_filename(filename, title_case=True)
    if "Audi" in base_name:
        keywords.insert(0, base_name)
        if "R8" in base_name and "GT" in base_name:
            keywords.insert(1, "Audi R8 GT")

    # Limit to top ~10 SEO phrases
    return ", ".join(keywords[:10])

def generate_seo_text(caption, filename):
    # Draft meta description
    meta_description = f"{caption}, showcasing luxury design, performance, and premium details."
    meta_description = adjust_meta_length(meta_description)

    # Keywords
    keywords = generate_keywords(caption, filename)

    return meta_description, keywords

# --- Streamlit UI ---
st.title("🖼 Bulk Image SEO Generator")
uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        caption = generate_caption(image, uploaded_file.name)
        meta_description, keywords = generate_seo_text(caption, uploaded_file.name)

        # Store structured SEO output
        seo_output = {
            "Filename": uploaded_file.name,
            "Alt attribute": caption,
            "Meta description": meta_description,
            "Keywords": keywords
        }
        results.append(seo_output)

        # Show in Streamlit UI
        st.subheader(f"📌 {uploaded_file.name}")
        st.markdown(f"**Alt attribute:** {caption}")
        st.markdown(f"**Meta description:** {meta_description}")
        st.markdown(f"**Keywords:** {keywords}")

    # Convert to DataFrame for CSV download
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="seo_image_data.csv", mime="text/csv")
