import os
import sys
# Streamlit Cloud fix for opencv-python pulling in GUI dependencies via ultralytics
os.system(f"{sys.executable} -m pip uninstall -y opencv-python opencv-contrib-python")
os.system(f"{sys.executable} -m pip install opencv-python-headless")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import io
import pandas as pd

from download_models import check_and_download_models
from pipeline import load_all_models, run_pipeline

# Page configuration
st.set_page_config(
    page_title="Receipt OCR & KIE Pipeline",
    layout="wide"
)

# Custom CSS for a clean, premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2563eb;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load models efficiently using cache
@st.cache_resource(show_spinner=False)
def load_models_cached():
    check_and_download_models()
    return load_all_models()

st.title("Receipt OCR & Key Information Extraction")
st.markdown("End-to-end pipeline: Segmentation, Normalization, Text Detection, OCR, and Key Information Extraction.")

with st.spinner("Initializing models..."):
    models = load_models_cached()

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from uploader
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("Failed to load image. Please upload a valid image file.")
    else:
        # Run pipeline
        with st.spinner("Processing image through pipeline..."):
            results = run_pipeline(models, img_bgr)
            
        st.success("Processing complete!")
        
        # Display Intermediate Steps
        st.subheader("Processing Stages")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1. Raw Input**")
            st.image(cv2.cvtColor(results["raw_img"], cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col2:
            st.markdown("**2. Segmented & Cropped**")
            st.image(cv2.cvtColor(results["cropped"], cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with col3:
            st.markdown("**3. Normalized (Grayscale)**")
            st.image(results["normalized"], use_container_width=True, channels="GRAY")
            
        st.divider()
        
        # Display Final Extracted Information
        st.subheader("Extracted Key Information")
        extracted_data = results["extracted_json"]
        
        if extracted_data:
            col_table, col_json = st.columns([2, 1])
            
            with col_table:
                st.markdown("**Data Table**")
                # Convert dict to dataframe for nice display
                df = pd.DataFrame(list(extracted_data.items()), columns=["Field", "Value"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
            with col_json:
                st.markdown("**JSON Output**")
                st.json(extracted_data)
                
                # Download button for JSON
                json_str = json.dumps(extracted_data, ensure_ascii=False, indent=4)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="extracted_receipt.json",
                    mime="application/json"
                )
        else:
            st.info("No text was confidently extracted from the receipt.")
