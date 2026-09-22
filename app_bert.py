# app_bert.py - Dashboard using BERT model
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

st.set_page_config(page_title="Mental Health Risk Detector - BERT", page_icon="🧠", layout="wide")

# Load BERT model
@st.cache_resource
def load_bert_model():
    model_path = 'results/BERT'
    if os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading BERT: {e}")
            return None, None
    return None, None

model, tokenizer = load_bert_model()

# Page
st.title("🧠 Mental Health Risk Detector (BERT Model)")
st.write("Using BERT-base-uncased — 83.6% accuracy")

if model is None:
    st.error("❌ BERT model not found. Run `python bert.py` first.")
    st.stop()

# Input
text = st.text_area("Enter text:", height=150)

if st.button("Analyze Risk", type="primary"):
    if text:
        with st.spinner("Analyzing with BERT..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            
            labels = ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
            pred_idx = np.argmax(probs)
            
            # Map to 3 risk levels
            risk_map = {
                'Normal': ('Low Risk', '#28A745'),
                'Depression': ('Moderate Risk', '#FFC107'),
                'Anxiety': ('Moderate Risk', '#FFC107'),
                'Stress': ('Moderate Risk', '#FFC107'),
                'Suicidal': ('High Risk', '#DC3545'),
                'Bipolar': ('High Risk', '#DC3545'),
                'Personality disorder': ('High Risk', '#DC3545')
            }
            
            risk_label, color = risk_map[labels[pred_idx]]
            confidence = probs[pred_idx] * 100
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 2rem; border-radius: 1rem; border: 3px solid {color}; text-align: center;">
                <h1 style="color: {color};">{risk_label}</h1>
                <p style="font-size: 1.3rem;">Confidence: <b>{confidence:.1f}%</b></p>
                <p style="font-size: 1rem;">Detected: <b>{labels[pred_idx]}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.subheader("📊 All Category Probabilities")
            prob_df = pd.DataFrame({'Category': labels, 'Probability': probs * 100})
            st.bar_chart(prob_df.set_index('Category'))
    else:
        st.warning("Please enter text.")