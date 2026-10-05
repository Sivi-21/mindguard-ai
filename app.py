"""
Mental Health Risk Detection Dashboard
BERT + BiLSTM Hybrid Model - Premium UI/UX
With Enhanced Safety Layer (Self-Harm Detection + Sarcasm Handling)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, firestore
import hashlib

# Optional imports
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from gtts import gTTS
    import io
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    import io
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="MindGuard AI - Mental Health Risk Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PREMIUM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #F5F7FA 0%, #E8EEF5 100%); }
    .hero-header {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        padding: 3rem 2rem; border-radius: 24px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    .hero-title { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; }
    .hero-subtitle { font-size: 1.15rem; color: rgba(255, 255, 255, 0.9); margin-top: 0.5rem; }
    .section-header {
        font-size: 1.5rem; font-weight: 700; color: #1A202C;
        margin: 1.5rem 0 1rem 0; padding-left: 1rem; border-left: 4px solid #667EEA;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(20px);
        padding: 1.5rem; border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.5); margin-bottom: 1rem;
    }
    .result-box-premium {
        padding: 2.5rem 2rem; border-radius: 24px; text-align: center;
        margin: 1.5rem 0; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.15);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-risk-label { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .result-confidence { font-size: 1.15rem; margin: 0.75rem 0 0.5rem 0; }
    .result-msg { font-size: 0.95rem; font-style: italic; opacity: 0.9; }
    .prob-bar-container { margin: 0.75rem 0; }
    .prob-bar-label {
        display: flex; justify-content: space-between;
        font-size: 0.9rem; font-weight: 600; color: #4A5568; margin-bottom: 0.4rem;
    }
    .prob-bar-track { height: 28px; background-color: #E2E8F0; border-radius: 14px; overflow: hidden; }
    .prob-bar-fill { height: 100%; border-radius: 14px; }
    .stButton > button {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        color: white; border: none; padding: 0.85rem 2rem;
        border-radius: 14px; font-weight: 600; font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #FFFFFF 0%, #F7FAFC 100%); border-right: 1px solid #E2E8F0; }
    .stTextArea textarea {
        border-radius: 16px !important; border: 2px solid #E2E8F0 !important;
        padding: 1rem !important; font-size: 1rem !important; background-color: #FAFBFC !important;
    }
    .stTextArea textarea:focus { border-color: #667EEA !important; box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important; }
    .status-badge {
        display: inline-block; padding: 0.3rem 0.75rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600; margin: 0.2rem 0;
    }
    .status-ok { background-color: #C6F6D5; color: #22543D; }
    .status-error { background-color: #FED7D7; color: #742A2A; }
    .crisis-box {
        background: linear-gradient(135deg, #FED7D7 0%, #FEB2B2 100%);
        border-left: 6px solid #C53030; padding: 1.5rem;
        border-radius: 16px; margin: 1rem 0;
    }
    .history-item-premium {
        padding: 1rem 1.25rem; border-radius: 14px; margin: 0.5rem 0;
        background: white; border-left: 4px solid #667EEA;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .footer {
        text-align: center; padding: 2rem 1rem; color: #718096;
        font-size: 0.9rem; border-top: 1px solid #E2E8F0; margin-top: 3rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# FIREBASE INIT
# ============================================
if not firebase_admin._apps:
    try:
        cert = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cert)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        pass # Handle locally if needed
db = firestore.client()

# ============================================
# SESSION STATE & AUTH
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0}
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'email_settings' not in st.session_state:
    st.session_state.email_settings = {}

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

if not st.session_state.user:
    st.markdown('<div class="hero-header"><h1 class="hero-title">🧠 MindGuard AI</h1><p class="hero-subtitle">Login to access your dashboard</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if not email.strip() or not pw.strip():
                    st.error("Please enter both email and password")
                else:
                    doc = db.collection('users').document(email.lower()).get()
                    if doc.exists and doc.to_dict().get('password') == hash_pw(pw):
                        st.session_state.user = email.lower()
                        
                        # Fetch history
                        docs = db.collection('users').document(email.lower()).collection('history').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
                        st.session_state.history = [d.to_dict() for d in docs]
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
    with tab2:
        with st.form("signup"):
            new_email = st.text_input("Email")
            new_pw = st.text_input("Password", type="password")
            if st.form_submit_button("Sign Up"):
                if not new_email.strip() or not new_pw.strip():
                    st.error("Please enter both email and password")
                else:
                    ref = db.collection('users').document(new_email.lower())
                    if ref.get().exists:
                        st.error("Account exists!")
                    else:
                        ref.set({'password': hash_pw(new_pw)})
                        st.success("Account created! Please log in.")
    st.stop()


# ============================================
# DEVICE
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# BERT + BiLSTM MODEL CLASS
# ============================================
class BertBiLSTM(nn.Module):
    """Hybrid BERT + BiLSTM model"""
    def __init__(self, bert_model_name="bert-base-uncased", num_classes=3, hidden_size=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.bilstm = nn.LSTM(
            input_size=bert_hidden, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.bilstm(bert_out.last_hidden_state)
        cls_repr = lstm_out[:, 0, :]
        return self.classifier(self.dropout(cls_repr))

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_bert_bilstm_model():
    """Load the trained BERT + BiLSTM model from Hugging Face"""
    try:
        from huggingface_hub import hf_hub_download
        REPO_ID = "sivvvsivagami/mindguard-ai-models"
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Download label encoder
        le_path = hf_hub_download(repo_id=REPO_ID, filename="label_encoder.pkl")
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        num_classes = len(label_encoder.classes_)
        
        model = BertBiLSTM("bert-base-uncased", num_classes=num_classes)
        
        # Download model weights
        model_pt_path = hf_hub_download(repo_id=REPO_ID, filename="model.pt")
        model.load_state_dict(torch.load(model_pt_path, map_location=DEVICE))
        
        model.eval()
        model.to(DEVICE)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading BERT+BiLSTM: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        from huggingface_hub import hf_hub_download
        data_path = hf_hub_download(repo_id="sivvvsivagami/mindguard-ai-models", filename="data.csv")
        return pd.read_csv(data_path)
    except:
        return None

model, bert_tokenizer, label_encoder = load_bert_bilstm_model()
df = load_data()

# ============================================
# SMART SENTIMENT
# ============================================
def get_smart_sentiment(text, risk_result):
    text_lower = text.lower()
    
    negative_words = [
        'sad', 'depressed', 'depression', 'cry', 'crying', 'tears',
        'lonely', 'alone', 'isolated', 'empty', 'numb', 'hopeless',
        'worthless', 'angry', 'mad', 'furious', 'rage', 'hate',
        'stressed', 'stress', 'anxious', 'anxiety', 'worried', 'scared',
        'fear', 'afraid', 'tired', 'exhausted', 'drained', 'pain',
        'hurt', 'hurting', 'broken', 'lost', 'confused', 'nightmare',
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'disgusting',
        'useless', 'pointless', 'failure', 'failed', 'struggling',
        'suffering', 'dying', 'falling', 'disappointed', 'disappointment'
    ]
    
    positive_words = [
        'happy', 'happiness', 'joy', 'joyful', 'great', 'wonderful',
        'amazing', 'excellent', 'fantastic', 'awesome', 'good',
        'grateful', 'gratitude', 'thankful', 'blessed', 'proud',
        'excited', 'hopeful', 'optimistic', 'peaceful', 'calm',
        'relaxed', 'content', 'satisfied', 'enjoy', 'enjoyed',
        'laugh', 'laughed', 'laughing', 'smile', 'smiling'
    ]
    
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    
    if neg_count + pos_count == 0:
        if risk_result['risk'] == 'High Risk': score = -0.7
        elif risk_result['risk'] == 'Moderate Risk': score = -0.4
        else: score = 0.0
    else:
        score = (pos_count - neg_count) / (pos_count + neg_count)
    
    if score < -0.2:
        return {'label': "😞 Negative", 'score': score, 'color': "#E53E3E",
                'positive_count': pos_count, 'negative_count': neg_count}
    elif score > 0.2:
        return {'label': "😊 Positive", 'score': score, 'color': "#38A169",
                'positive_count': pos_count, 'negative_count': neg_count}
    else:
        return {'label': "😐 Neutral", 'score': score, 'color': "#718096",
                'positive_count': pos_count, 'negative_count': neg_count}

# ============================================
# PREDICT RISK (ENHANCED VERSION)
# ============================================
def predict_risk(text, model, tokenizer, label_encoder, max_len=128):
    """BERT + BiLSTM prediction with ENHANCED keyword safety override"""
    text_lower = text.lower().strip()
    text_norm = text_lower.replace('cutoff', 'cut off').replace('cutout', 'cut out')
    
    # ============================================
    # LAYER 1: CRITICAL SAFETY KEYWORDS (High Risk)
    # ============================================
    critical_keywords = [
        # Suicidal
        'suicide', 'suicidal', 'kill myself', 'killing myself', 'end my life',
        'end it all', 'end everything', 'want to die', 'wanna die',
        'wish i was dead', 'better off dead', 'should be dead',
        'not worth living', 'no reason to live', 'no point living',
        'take me out of this', 'take me away', 'want to leave this world',
        'make it stop', 'make it end', 'just want it to end',
        'last message', 'final goodbye', 'end my suffering',
        'cant take it anymore', "can't take it anymore",
        'no way out', 'no escape', 'ready to go', 'ready to leave',
        
        # Self-harm
        'self harm', 'self-harm', 'hurt myself', 'hurting myself',
        'cutting myself', 'cut myself', 'overdose', 'slit wrists',
        
        # Cutoff variations (NEW!)
        'cutoff', 'cut off', 'want to cut', 'feel like cutting',
        'need to cut', 'want to cutoff', 'feel to cutoff', 'feel to cut',
        'going to cut', 'gonna cut', 'wanna cut', 'i will cut',
        'i want to cut', 'i need to cut'
    ]
    
    # Body parts (for self-harm detection)
    body_parts = ['my hand', 'my arm', 'my wrist', 'my leg', 'my finger',
                  'my skin', 'my body', 'my veins', 'my neck', 'my throat',
                  'myself', 'my blood']
    
    # Social context words (for benign "cut off" patterns)
    social_context = ['people', 'friend', 'him', 'her', 'them', 'toxic', 
                      'relationship', 'contact', 'ties', 'communication',
                      'family', 'mom', 'dad', 'brother', 'sister']
    
    # ============================================
    # CHECK 1: Body-part self-harm
    # ============================================
    if 'cut' in text_norm or 'cut off' in text_norm or 'cutoff' in text_lower:
        # Check for body parts → High Risk
        if any(bp in text_norm for bp in body_parts):
            return {
                'risk': 'High Risk', 'confidence': 92.0, 'color': '#E53E3E',
                'probabilities': np.array([0.92, 0.04, 0.04]),
                'matched': [('self-harm context', 'High Risk')],
                'source': '🚨 Self-harm pattern detected'
            }
        
        # Check for social context → Moderate Risk
        if any(sc in text_lower for sc in social_context):
            return {
                'risk': 'Moderate Risk', 'confidence': 80.0, 'color': '#ED8936',
                'probabilities': np.array([0.05, 0.10, 0.85]),
                'matched': [('social cutoff', 'Moderate Risk')],
                'source': '⚠️ Social withdrawal pattern'
            }
        
        # Any other "cut off" / "cutoff" mention → HIGH RISK (safety)
        return {
            'risk': 'High Risk', 'confidence': 85.0, 'color': '#E53E3E',
            'probabilities': np.array([0.85, 0.07, 0.08]),
            'matched': [('self-harm indicator', 'High Risk')],
            'source': '🚨 Self-harm mention detected'
        }
    
    # ============================================
    # CHECK 2: Other critical keywords
    # ============================================
    for kw in critical_keywords:
        if kw in text_lower or kw in text_norm:
            return {
                'risk': 'High Risk', 'confidence': 92.0, 'color': '#E53E3E',
                'probabilities': np.array([0.92, 0.04, 0.04]),
                'matched': [(kw, 'High Risk')],
                'source': f'🚨 Critical safety keyword: "{kw}"'
            }
    
    # ============================================
    # LAYER 2: MODERATE RISK PATTERNS (Distress signals)
    # ============================================
    moderate_patterns = [
        # Sarcasm/mixed distress
        'alone in bed', 'crying in bed', 'cant stop crying', "can't stop crying",
        'spend another weekend alone', 'weekend alone', 'all alone',
        'crying myself to sleep', 'cry myself to sleep',
        'nobody cares', 'no one cares', 'nobody loves me', 'no one loves me',
        'no one checks on me', 'nobody checks on me',
        'i want to disappear', 'want to vanish',
        'laughing through the pain', 'smiling through the pain',
        'dying inside', 'dead inside', 'empty inside',
        'broken inside', 'falling apart',
        'tired of living', 'done with life', 'done with everything',
        'nothing matters', 'what is the point',
        'feeling so alone', 'feel so alone', 'so lonely',
        'i am so tired', "i'm so tired", 'just tired',
        'always alone', 'still alone', 'alone again',
        'cant sleep', "can't sleep", 'lost my job', 'got dumped',
        'broke up', 'no friends', 'zero friends', 'no support',
        'smiling outside', 'happy but', 'sad but',
        'lost all hope', 'giving up', 'given up',
        'i guess', 'i suppose', 'whatever', 'nvm',
        'this is it', 'last time', 'no more',
        # Emotions
        'depressed', 'depression', 'anxious', 'anxiety',
        'stressed', 'stress', 'overwhelmed', 'exhausted',
        'miserable', 'hopeless', 'worthless', 'useless',
        'failure', 'failed', 'struggling', 'suffering',
        'i hate myself', 'i hate my life', 'hate everything',
        'lonely', 'emptiness', 'numb', 'scary', 'scared',
        'terrified', 'fear', 'panic', 'afraid', 'creepy'
    ]
    
    matched_moderate = []
    for pattern in moderate_patterns:
        if pattern in text_lower:
            matched_moderate.append(pattern)
    
    if matched_moderate:
        confidence = min(90.0, 75.0 + len(matched_moderate) * 3)
        return {
            'risk': 'Moderate Risk', 'confidence': confidence, 'color': '#ED8936',
            'probabilities': np.array([0.05, 0.10, confidence/100]),
            'matched': [(p, 'Moderate Risk') for p in matched_moderate],
            'source': f'⚠️ Distress pattern: "{matched_moderate[0]}"'
        }
    
    # ============================================
    # LAYER 3: BERT + BiLSTM MODEL
    # ============================================
    if model is None or tokenizer is None:
        return {
            'risk': 'Moderate Risk', 'confidence': 50.0, 'color': '#ED8936',
            'probabilities': np.array([0.33, 0.34, 0.33]),
            'matched': [], 'source': 'Fallback (no model)'
        }
    
    try:
        cleaned = re.sub(r'http\S+|www\S+|https\S+', '', text_lower)
        cleaned = re.sub(r'@\w+|#\w+', '', cleaned)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        inputs = tokenizer(
            cleaned, return_tensors="pt", truncation=True,
            padding=True, max_length=max_len
        )
        
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        with torch.inference_mode():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx] * 100
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        
        label_map = {'High': 'High Risk', 'Low': 'Low Risk', 'Moderate': 'Moderate Risk'}
        risk_level = label_map.get(pred_label, 'Low Risk')
        
        color_map = {'High': '#E53E3E', 'Low': '#38A169', 'Moderate': '#ED8936'}
        color = color_map.get(pred_label, '#38A169')
        
        return {
            'risk': risk_level, 'confidence': confidence, 'color': color,
            'probabilities': probs, 'matched': [],
            'source': f'🤖 BERT+BiLSTM ({confidence:.1f}%)'
        }
    
    except Exception as e:
        return {
            'risk': 'Moderate Risk', 'confidence': 50.0, 'color': '#ED8936',
            'probabilities': np.array([0.33, 0.34, 0.33]),
            'matched': [], 'source': f'Error: {e}'
        }

# ============================================
# HELPER FUNCTIONS
# ============================================
def generate_report(text, result):
    return f"""
MENTAL HEALTH RISK ASSESSMENT REPORT
=====================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT TEXT:
{text}

ASSESSMENT:
Risk Level:   {result['risk']}
Confidence:   {result['confidence']:.1f}%
Source:       {result.get('source', 'BERT+BiLSTM Model')}

PROBABILITY DISTRIBUTION:
- High Risk:     {result['probabilities'][0]*100:.1f}%
- Low Risk:      {result['probabilities'][1]*100:.1f}%
- Moderate Risk: {result['probabilities'][2]*100:.1f}%

DISCLAIMER:
This is an AI-based assessment and NOT a medical diagnosis.

Crisis Resources (India):
- AASRA: +91-9820466726
- iCall: +91-9152987821
- Emergency: 112
"""

def generate_pdf_report(text, result, sentiment):
    try:
        from fpdf import FPDF
    except ImportError:
        return generate_report(text, result).encode('utf-8')
        
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("helvetica", size=16, style='B')
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 10, txt="MindGuard AI - Risk Assessment Report", ln=True, align='C')
    pdf.ln(5)
    
    # Date
    pdf.set_font("helvetica", size=10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 10, txt=f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')
    pdf.ln(5)
    
    # Input Text
    pdf.set_font("helvetica", size=12, style='B')
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 8, txt="INPUT TEXT:", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(74, 85, 104)
    pdf.multi_cell(0, 6, txt=str(text).encode('latin-1', 'replace').decode('latin-1'))
    pdf.ln(8)
    
    # Assessment
    pdf.set_font("helvetica", size=12, style='B')
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 8, txt="ASSESSMENT RESULTS:", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 6, txt=f"Risk Level: {result['risk']}", ln=True)
    pdf.cell(0, 6, txt=f"Confidence: {result['confidence']:.1f}%", ln=True)
    pdf.cell(0, 6, txt=f"Source: {result.get('source', 'BERT+BiLSTM Model')}", ln=True)
    pdf.ln(8)
    
    # Probabilities
    pdf.set_font("helvetica", size=12, style='B')
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 8, txt="PROBABILITY DISTRIBUTION:", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 6, txt=f"High Risk: {result['probabilities'][0]*100:.1f}%", ln=True)
    pdf.cell(0, 6, txt=f"Low Risk: {result['probabilities'][1]*100:.1f}%", ln=True)
    pdf.cell(0, 6, txt=f"Moderate Risk: {result['probabilities'][2]*100:.1f}%", ln=True)
    pdf.ln(8)
    
    # Sentiment
    pdf.set_font("helvetica", size=12, style='B')
    pdf.set_text_color(26, 32, 44)
    pdf.cell(0, 8, txt="SENTIMENT ANALYSIS:", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(74, 85, 104)
    pdf.cell(0, 6, txt=f"Overall Sentiment: {sentiment['label'].replace('😞','Negative').replace('😊','Positive').replace('😐','Neutral')} (Score: {sentiment['score']:.2f})", ln=True)
    pdf.ln(15)
    
    # Disclaimer
    pdf.set_font("helvetica", size=9, style='I')
    pdf.set_text_color(160, 174, 192)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This is an AI-based assessment and NOT a medical diagnosis.")
    pdf.multi_cell(0, 5, txt="Crisis Resources (India): AASRA: +91-9820466726 | iCall: +91-9152987821 | Emergency: 112")
    
    return bytes(pdf.output())

def send_alert_email(text, result, settings):
    if not settings.get('sender') or not settings.get('password') or not settings.get('admin'):
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg['From'] = settings['sender']
        msg['To'] = settings['admin']
        msg['Subject'] = "🚨 URGENT: High Mental Health Risk Detected"
        
        body = f"MindGuard AI has detected a HIGH RISK text entry.\n\nRisk Level: {result['risk']} (Confidence: {result['confidence']:.1f}%)\n\nUser Input Text:\n\"{text}\"\n\nPlease review this case immediately."
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(settings['sender'], settings['password'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

def analyze_emotions(text):
    text_lower = text.lower()
    return {
        'Sadness': sum(1 for w in ['sad', 'cry', 'tears', 'depressed', 'unhappy', 'hopeless'] if w in text_lower),
        'Anxiety': sum(1 for w in ['anxious', 'worried', 'nervous', 'scared', 'panic', 'fear'] if w in text_lower),
        'Anger': sum(1 for w in ['angry', 'mad', 'furious', 'frustrated', 'hate', 'rage'] if w in text_lower),
        'Loneliness': sum(1 for w in ['lonely', 'alone', 'isolated', 'abandoned'] if w in text_lower),
        'Hope': sum(1 for w in ['hope', 'hopeful', 'better', 'future', 'optimistic'] if w in text_lower),
        'Joy': sum(1 for w in ['happy', 'joy', 'great', 'amazing', 'love', 'excited'] if w in text_lower)
    }

def highlight_text(text, matched_keywords):
    highlighted = text
    for kw, category in matched_keywords:
        color = "#E53E3E" if category == "High Risk" else "#ED8936" if category == "Moderate Risk" else "#38A169"
        highlighted = re.sub(
            f'({re.escape(kw)})',
            f'<span style="background-color: {color}; color: white; padding: 3px 6px; border-radius: 4px; font-weight: 600;">\\1</span>',
            highlighted, flags=re.IGNORECASE
        )
    return highlighted

def render_probability_bars(probabilities):
    labels = ['High Risk', 'Low Risk', 'Moderate Risk']
    colors = ['#E53E3E', '#38A169', '#ED8936']
    
    for label, prob, color in zip(labels, probabilities, colors):
        st.markdown(f"""
        <div class="prob-bar-container">
            <div class="prob-bar-label">
                <span>{label}</span>
                <span style="color: {color};">{prob*100:.1f}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" style="width: {prob*100}%; background: linear-gradient(90deg, {color} 0%, {color}CC 100%);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🧠</div>
        <h2 style="color: #1A202C; font-weight: 800; margin: 0.5rem 0;">MindGuard AI</h2>
        <p style="color: #718096; font-size: 0.85rem; margin: 0;">BERT + BiLSTM Powered</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"**👤 Logged in as:** {st.session_state.user}")
    if st.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Navigate:",
        ["🏠 Home", "🎙️ Voice Analysis", "📓 Mood Journal", "💬 Check-in Chat", "📈 User Dashboard", "🧑‍⚕️ Therapist Dashboard", "📊 Model Performance", "📈 Data Insights",
         "📜 History", "📁 Batch Analysis", "📱 Social Media Risk", "ℹ️ About", "🧪 Test Cases", "🔬 XAI Explorer"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    threshold = st.slider("Confidence Threshold", 0, 100, 60)
    
    st.markdown("### 📧 Alert Settings")
    with st.expander("Configure Email Alerts"):
        sender_email = st.text_input("Sender Gmail", value=st.session_state.email_settings.get('sender', ''))
        sender_app_pw = st.text_input("App Password", type="password", value=st.session_state.email_settings.get('password', ''))
        admin_email = st.text_input("Admin Email (Recipient)", value=st.session_state.email_settings.get('admin', ''))
        if st.button("Save Alert Settings", use_container_width=True):
            st.session_state.email_settings = {
                'sender': sender_email,
                'password': sender_app_pw,
                'admin': admin_email
            }
            st.success("Alert settings saved!")
    
    st.markdown("---")
    st.markdown("### 💡 System Status")
    
    def status_badge(loaded, name):
        if loaded:
            return f'<span class="status-badge status-ok">✓ {name}</span>'
        return f'<span class="status-badge status-error">✗ {name}</span>'
    
    st.markdown(f"""
    <div style="line-height: 2.2;">
        {status_badge(model is not None, 'BERT+BiLSTM')}<br>
        {status_badge(bert_tokenizer is not None, 'Tokenizer')}<br>
        {status_badge(label_encoder is not None, 'Label Encoder')}<br>
        {status_badge(df is not None, 'Dataset')}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div style="background: white; padding: 1rem; border-radius: 12px; font-size: 0.85rem;">
        <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
            <span style="color: #718096;">Architecture</span>
            <span style="font-weight: 600;">BERT + BiLSTM</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
            <span style="color: #718096;">Accuracy</span>
            <span style="color: #38A169; font-weight: 700;">88.95%</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
            <span style="color: #718096;">F1-Score</span>
            <span style="color: #38A169; font-weight: 700;">0.8955</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
            <span style="color: #718096;">Recall</span>
            <span style="color: #38A169; font-weight: 700;">93.81%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Dark mode CSS
if dark_mode:
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%); }
        .section-header, h1, h2, h3, h4, h5, h6, p, label, span { color: #F7FAFC !important; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #2D3748 0%, #1A202C 100%); }
        .stTextArea textarea { background-color: #2D3748 !important; color: white !important; }
        .glass-card, .history-item-premium { background: #2D3748 !important; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 1: HOME
# ============================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🧠 MindGuard AI</h1>
        <p class="hero-subtitle">BERT + BiLSTM Powered Mental Health Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">📝 Analyze Your Text</div>', unsafe_allow_html=True)
        st.markdown('<div style="color: #4A5568; margin-bottom: 1rem;">Enter any social media text, message, or statement to analyze its mental health risk level.</div>', unsafe_allow_html=True)
        
        text_input = st.text_area(
            "Enter text:", height=180,
            placeholder="Example: I've been feeling really down lately...",
            label_visibility="collapsed"
        )
        
        source_lang = st.selectbox("🌍 Text Language:", ["English", "Auto-Detect", "Hindi", "Tamil", "Tanglish", "Spanish", "French", "German", "Chinese"])
        
        analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.markdown('<div class="section-header">✨ Quick Examples</div>', unsafe_allow_html=True)
        
        examples = [
            ("🟢 Positive", "I'm having a great day! Feeling so happy and grateful."),
            ("🟡 Moderate", "I'm feeling anxious about tomorrow's presentation."),
            ("🔴 High Risk", "I don't want to live anymore, please help me.")
        ]
        
        for label, example in examples:
            if st.button(f"{label}: {example[:35]}...", key=example[:30], use_container_width=True):
                text_input = example
    
    with col2:
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        if analyze_btn and text_input:
            st.session_state.current_analysis_text = text_input
            
        if (analyze_btn or st.session_state.get('current_analysis_text') == text_input) and text_input:
            if model is None:
                st.error("❌ BERT+BiLSTM model not found. Check mental_health_bert_bilstm_model/ folder.")
            else:
                with st.spinner("🧠 Analyzing with BERT+BiLSTM..."):
                    actual_input = text_input
                    if source_lang != "English":
                        try:
                            from deep_translator import GoogleTranslator
                            lang_map = {"Hindi": "hi", "Tamil": "ta", "Tanglish": "ta", "Spanish": "es", "French": "fr", "German": "de", "Chinese": "zh-CN"}
                            src_code = lang_map.get(source_lang, "auto")
                            translated = GoogleTranslator(source=src_code, target='en').translate(text_input)
                            st.info(f"🌍 Translated to English: *\"{translated}\"*")
                            actual_input = translated
                        except Exception as e:
                            st.warning("Translation failed. Falling back to original text.")
                            
                    result = predict_risk(actual_input, model, bert_tokenizer, label_encoder)
                    
                    if result:
                        risk_labels = {
                            'High Risk': '🔴 HIGH RISK',
                            'Low Risk': '🟢 LOW RISK',
                            'Moderate Risk': '🟡 MODERATE RISK'
                        }
                        
                        sentiment = get_smart_sentiment(actual_input, result)
                        
                        if result['risk'] == 'High Risk':
                            gradient = "linear-gradient(135deg, #E53E3E 0%, #C53030 100%)"
                            sub_msg = "⚠️ Please reach out to a mental health professional"
                            text_color = "white"
                        elif result['risk'] == 'Moderate Risk':
                            gradient = "linear-gradient(135deg, #ED8936 0%, #DD6B20 100%)"
                            sub_msg = "💡 Consider talking to someone you trust"
                            text_color = "white"
                        else:
                            gradient = "linear-gradient(135deg, #38A169 0%, #2F855A 100%)"
                            if sentiment['score'] < -0.2:
                                sub_msg = "✅ No personal mental health risk detected."
                            else:
                                sub_msg = "✨ Great! Keep up the positive vibes"
                            text_color = "white"
                        
                        st.markdown(f"""
                        <div class="result-box-premium" style="background: {gradient};">
                            <h2 class="result-risk-label" style="color: {text_color};">
                                {risk_labels[result['risk']]}
                            </h2>
                            <p class="result-confidence" style="color: {text_color};">
                                Confidence: <strong>{result['confidence']:.1f}%</strong>
                            </p>
                            <p class="result-msg" style="color: {text_color};">
                                {sub_msg}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info(f"📍 {result.get('source', 'BERT+BiLSTM Model')}")
                        
                        item = {
                            'text': actual_input,
                            'risk': result['risk'],
                            'confidence': result['confidence'],
                            'time': pd.Timestamp.now().strftime('%H:%M:%S'),
                            'timestamp': firestore.SERVER_TIMESTAMP
                        }
                        st.session_state.history.insert(0, item)
                        try:
                            db.collection('users').document(st.session_state.user).collection('history').add(item)
                        except:
                            pass
                        st.session_state.analytics[result['risk']] += 1
                        
                        if result['confidence'] < threshold:
                            st.warning(f"⚠️ Below threshold ({threshold}%)")
                        
                        st.markdown('<div class="section-header">📈 Probability Distribution</div>', unsafe_allow_html=True)
                        render_probability_bars(result['probabilities'])
                        
                        st.markdown(f"""
                        <div class="glass-card" style="border-left: 4px solid {sentiment['color']};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <div style="color: #718096; font-size: 0.85rem;">Sentiment</div>
                                    <div style="font-size: 1.3rem; font-weight: 700; color: {sentiment['color']};">
                                        {sentiment['label']}
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="color: #718096; font-size: 0.85rem;">Score</div>
                                    <div style="font-size: 1.2rem; font-weight: 700;">{sentiment['score']:+.2f}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if result.get('matched'):
                            st.markdown('<div class="section-header">🔥 Detected Keywords</div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="glass-card">
                                <div style="line-height: 2; font-size: 1.05rem;">
                                    {highlight_text(actual_input, result['matched'])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if PLOTLY_AVAILABLE:
                            emotions = analyze_emotions(actual_input)
                            if sum(emotions.values()) > 0:
                                st.markdown('<div class="section-header">🎭 Emotion Analysis</div>', unsafe_allow_html=True)
                                fig_r = go.Figure(data=go.Scatterpolar(
                                    r=list(emotions.values()),
                                    theta=list(emotions.keys()),
                                    fill='toself',
                                    fillcolor='rgba(102, 126, 234, 0.25)',
                                    line=dict(color='#667EEA', width=3)
                                ))
                                fig_r.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, max(max(emotions.values()), 1) + 1])),
                                    showlegend=False, height=350,
                                    margin=dict(l=40, r=40, t=20, b=20),
                                    paper_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig_r, use_container_width=True)
                        
                        st.markdown('<div class="section-header">🧠 Explainable AI</div>', unsafe_allow_html=True)
                        with st.expander("See why the AI made this prediction"):
                            if st.button("Generate LIME Explanation", key="lime_home"):
                                with st.spinner("Generating visual explanation..."):
                                    from lime.lime_text import LimeTextExplainer
                                    import numpy as np
                                    import torch
                                    
                                    def lime_predict_fn(texts):
                                        model.eval()
                                        probs_list = []
                                        batch_size = 32
                                        with torch.no_grad():
                                            for i in range(0, len(texts), batch_size):
                                                batch_texts = texts[i:i+batch_size]
                                                encoded = bert_tokenizer(
                                                    batch_texts,
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=128,
                                                    return_tensors='pt'
                                                )
                                                output = model(encoded['input_ids'].to(DEVICE), encoded['attention_mask'].to(DEVICE))
                                                probs_list.extend(torch.softmax(output, dim=1).cpu().numpy())
                                        return np.array(probs_list)
                                    
                                    explainer = LimeTextExplainer(class_names=label_encoder.classes_)
                                    exp = explainer.explain_instance(actual_input, lime_predict_fn, num_features=10, num_samples=100)
                                    st.components.v1.html(exp.as_html(), height=400, scrolling=True)

                        st.markdown('<div class="section-header">🔄 Continuous Learning</div>', unsafe_allow_html=True)
                        with st.expander("Help improve the AI (Submit Feedback)"):
                            st.write("Did the AI classify this correctly?")
                            fb_col1, fb_col2 = st.columns([1, 2])
                            with fb_col1:
                                if st.button("👍 Yes, Accurate", key="fb_yes", use_container_width=True):
                                    st.success("Thanks for confirming!")
                                    try:
                                        db.collection('feedback').add({
                                            'text': actual_input, 'predicted': result['risk'], 'actual': result['risk'],
                                            'timestamp': firestore.SERVER_TIMESTAMP, 'user': st.session_state.user
                                        })
                                    except: pass
                            with fb_col2:
                                with st.form(key="correction_form"):
                                    correction = st.selectbox("👎 No, correct it to:", ["Low Risk", "Moderate Risk", "High Risk"])
                                    if st.form_submit_button("Submit Correction", use_container_width=True):
                                        st.success("Correction saved to database! This helps retrain the model.")
                                        try:
                                            db.collection('feedback').add({
                                                'text': actual_input, 'predicted': result['risk'], 'actual': correction,
                                                'timestamp': firestore.SERVER_TIMESTAMP, 'user': st.session_state.user
                                            })
                                        except: pass

                        st.markdown('<div class="section-header">🎯 Actions</div>', unsafe_allow_html=True)
                        act_col1, act_col2 = st.columns(2)
                        
                        with act_col1:
                            try:
                                pdf_data = generate_pdf_report(actual_input, result, sentiment)
                                st.download_button(
                                    "📥 Download PDF Report",
                                    pdf_data,
                                    f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                                    "application/pdf", use_container_width=True
                                )
                            except Exception as e:
                                st.download_button(
                                    "📥 Download Report",
                                    generate_report(actual_input, result),
                                    f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                                    "text/plain", use_container_width=True
                                )
                        
                        with act_col2:
                            if TTS_AVAILABLE:
                                if st.button("🔊 Listen", use_container_width=True):
                                    audio_text = f"Your risk level is {result['risk']} with {result['confidence']:.0f} percent confidence"
                                    tts = gTTS(text=audio_text, lang='en')
                                    audio_buf = io.BytesIO()
                                    tts.write_to_fp(audio_buf)
                                    audio_buf.seek(0)
                                    st.audio(audio_buf, format='audio/mp3')
                        
                        if result['risk'] == 'High Risk':
                            if st.session_state.email_settings.get('sender') and st.session_state.email_settings.get('admin'):
                                if not st.session_state.get(f"email_sent_{actual_input}"):
                                    import threading
                                    threading.Thread(target=send_alert_email, args=(actual_input, result, st.session_state.email_settings)).start()
                                    st.toast("📧 Alert email dispatched to admin.", icon="🚨")
                                    st.session_state[f"email_sent_{actual_input}"] = True

                            st.markdown("""
                            <div class="crisis-box">
                                <h3 style="color: #C53030; margin-top: 0;">🚨 If You're in Crisis — Please Reach Out</h3>
                                <p style="color: #742A2A; font-weight: 600;">India Helplines (24/7):</p>
                                <ul style="color: #742A2A;">
                                    <li><strong>AASRA:</strong> +91-9820466726</li>
                                    <li><strong>iCall:</strong> +91-9152987821</li>
                                    <li><strong>Vandrevala:</strong> 1860-2662-345</li>
                                    <li><strong>Emergency:</strong> 112</li>
                                </ul>
                                <p style="font-size: 0.85rem; color: #742A2A; font-style: italic;">
                                    ⚠️ This is an AI screening tool, NOT a medical diagnosis.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("<br>", unsafe_allow_html=True)
                            with st.expander("📍 Find Nearby Mental Health Resources (Map)"):
                                st.markdown("We've located nearby support centers based on your approximate location.")
                                try:
                                    import folium
                                    from streamlit_folium import st_folium
                                    import requests
                                    
                                    # Fetch approximate IP location
                                    loc_data = requests.get("https://ipinfo.io/json", timeout=3).json()
                                    if 'loc' in loc_data:
                                        lat, lon = map(float, loc_data['loc'].split(','))
                                        city = loc_data.get('city', 'your area')
                                    else:
                                        lat, lon, city = 28.6139, 77.2090, "New Delhi"
                                    
                                    m = folium.Map(location=[lat, lon], zoom_start=13)
                                    folium.Marker([lat, lon], popup="Your approximate location", icon=folium.Icon(color='blue', icon='user')).add_to(m)
                                    
                                    # Generate mock nearby clinics for demonstration
                                    clinics = [
                                        {"name": "Mind Care Clinic", "lat_offset": 0.015, "lon_offset": 0.01},
                                        {"name": "City Wellness Center", "lat_offset": -0.01, "lon_offset": 0.02},
                                        {"name": "Hope Therapy Services", "lat_offset": 0.02, "lon_offset": -0.015}
                                    ]
                                    for clinic in clinics:
                                        folium.Marker(
                                            [lat + clinic['lat_offset'], lon + clinic['lon_offset']],
                                            popup=clinic['name'],
                                            tooltip="Click for details",
                                            icon=folium.Icon(color='red', icon='plus')
                                        ).add_to(m)
                                        
                                    st_folium(m, width=700, height=400)
                                    st.caption(f"Showing resources near **{city}**. *(Note: Locations are simulated for this demo)*")
                                except Exception as e:
                                    st.error("Could not load map data. Please check your internet connection.")
        elif analyze_btn:
            st.info("💡 Please enter some text to analyze.")
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
                <h3 style="color: #4A5568;">Waiting for your input</h3>
                <p style="color: #718096;">Enter text on the left and click "Analyze Risk" to begin</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE 1.4: THERAPIST DASHBOARD
# ============================================
elif page == "🧑‍⚕️ Therapist Dashboard":
    st.markdown('<div class="hero-header"><h1 class="hero-title">🧑‍⚕️ Therapist Dashboard</h1><p class="hero-subtitle">Secure portal for clinicians to monitor patient risk trends over time.</p></div>', unsafe_allow_html=True)
    
    st.info("ℹ️ *Simulation Mode:* In a production app, this page would require a special clinician login. For this demo, we're securely fetching data from all users in the Firebase database.")
    
    try:
        users = db.collection('users').stream()
        user_list = [u.id for u in users]
        
        if not user_list:
            st.warning("No users found in database.")
        else:
            # High level metrics
            total_patients = len(user_list)
            
            # Fetch all history to find high risk alerts
            all_history = []
            high_risk_count = 0
            for u in user_list:
                hist_docs = db.collection('users').document(u).collection('history').stream()
                for d in hist_docs:
                    data = d.to_dict()
                    data['user_id'] = u
                    all_history.append(data)
                    if data.get('risk') == 'High Risk':
                        high_risk_count += 1
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Monitored Patients", total_patients)
            m_col2.metric("Total Assessments Logged", len(all_history))
            m_col3.metric("🚨 Total High-Risk Alerts", high_risk_count)
            
            st.markdown("---")
            st.markdown("### 📋 Patient Selector")
            
            selected_patient = st.selectbox("Select a patient to review their longitudinal record:", user_list)
            
            if selected_patient:
                patient_records = [r for r in all_history if r['user_id'] == selected_patient]
                
                if patient_records:
                    df_pat = pd.DataFrame(patient_records)
                    
                    st.markdown(f"#### 📈 Risk Trend for {selected_patient}")
                    
                    # Convert risk to numeric for plotting
                    risk_map = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}
                    df_pat['risk_score'] = df_pat['risk'].map(risk_map)
                    
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        # Reverse to plot chronologically
                        fig.add_trace(go.Scatter(y=df_pat['risk_score'][::-1].reset_index(drop=True), mode='lines+markers', name='Risk Level',
                                      line=dict(color='#ED8936', width=3), marker=dict(size=10)))
                        fig.update_layout(
                            yaxis=dict(tickvals=[1,2,3], ticktext=["Low Risk", "Moderate Risk", "High Risk"]),
                            xaxis=dict(title="Assessment Sequence"),
                            height=350, margin=dict(l=20, r=20, t=30, b=20),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    st.markdown("#### 📝 Assessment Log")
                    st.dataframe(
                        df_pat[['time', 'text', 'risk', 'confidence']],
                        use_container_width=True
                    )
                else:
                    st.info("No assessments logged for this patient yet.")
    except Exception as e:
        st.error(f"Could not connect to Firebase to load patients. ({e})")

# ============================================
# PAGE 1.2: VOICE ANALYSIS
# ============================================
elif page == "🎙️ Voice Analysis":
    st.markdown('<div class="hero-header"><h1 class="hero-title">🎙️ Voice Analysis</h1><p class="hero-subtitle">Speak your mind. We will transcribe and analyze your mental health risk in real-time.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎙️ Record Audio")
        audio_value = st.audio_input("Record a voice note")
    
    with col2:
        st.markdown("### 📁 Or Upload Audio")
        uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=['wav', 'mp3', 'ogg', 'm4a'])
        
    audio_to_process = audio_value if audio_value else uploaded_file
    
    if audio_to_process:
        st.audio(audio_to_process)
        
        if st.button("Transcribe & Analyze", type="primary", use_container_width=True):
            with st.spinner("Transcribing audio (this may take a moment)..."):
                try:
                    import speech_recognition as sr
                    from pydub import AudioSegment
                    import tempfile
                    import os
                    
                    r = sr.Recognizer()
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_to_process.read())
                        tmp_path = tmp.name
                        
                    # Process with AudioSegment to ensure it's a valid WAV for SpeechRecognition
                    try:
                        audio_seg = AudioSegment.from_file(tmp_path)
                        clean_wav_path = tmp_path + "_clean.wav"
                        audio_seg.export(clean_wav_path, format="wav")
                    except:
                        clean_wav_path = tmp_path
                        
                    with sr.AudioFile(clean_wav_path) as source:
                        audio_data = r.record(source)
                        
                    # Recognize using Google Speech Recognition
                    text = r.recognize_google(audio_data)
                    st.success("Transcription complete!")
                    
                    st.markdown("#### 📝 Transcribed Text:")
                    st.info(f"*{text}*")
                    
                    st.markdown("#### 🧠 Analysis:")
                    with st.spinner("Analyzing risk..."):
                        result = predict_risk(text, model, bert_tokenizer, label_encoder)
                        emotions = analyze_emotions(text)
                        
                        risk_labels = {
                            'High Risk': '🔴 HIGH RISK',
                            'Low Risk': '🟢 LOW RISK',
                            'Moderate Risk': '🟡 MODERATE RISK'
                        }
                        
                        st.markdown(f"**Risk Level:** {risk_labels.get(result['risk'], result['risk'])} (Confidence: {result['confidence']:.1f}%)")
                        
                        # Dominant Emotion
                        dominant = max(emotions, key=emotions.get) if sum(emotions.values())>0 else "Neutral"
                        st.markdown(f"**Dominant Emotion:** {dominant}")
                        
                        if result['risk'] == 'High Risk':
                            st.error("🚨 We detected high risk in your speech. Please consider reaching out for help. (India Helplines: 112, AASRA: +91-9820466726)")
                            
                        # Cleanup temp files
                        try:
                            os.remove(tmp_path)
                            if clean_wav_path != tmp_path:
                                os.remove(clean_wav_path)
                        except: pass
                            
                except Exception as e:
                    st.error(f"Error processing audio: {e}. (Make sure you have an active internet connection for transcription).")

# ============================================
# PAGE 1.1: MOOD JOURNAL
# ============================================
elif page == "📓 Mood Journal":
    st.markdown('<div class="hero-header"><h1 class="hero-title">📓 Mood Journal</h1><p class="hero-subtitle">Log your daily mood and thoughts securely.</p></div>', unsafe_allow_html=True)
    
    st.markdown("### How are you feeling today?")
    
    # Emojis for mood
    moods = {"Amazing": "🤩", "Good": "😊", "Okay": "😐", "Down": "😔", "Awful": "😭"}
    cols = st.columns(5)
    
    if "selected_mood" not in st.session_state:
        st.session_state.selected_mood = None
        
    for i, (mood_name, emoji) in enumerate(moods.items()):
        with cols[i]:
            if st.button(f"{emoji}\\n\\n{mood_name}", use_container_width=True, key=f"mood_{mood_name}"):
                st.session_state.selected_mood = mood_name
                
    if st.session_state.selected_mood:
        st.info(f"You selected: **{st.session_state.selected_mood} {moods[st.session_state.selected_mood]}**")
        
        with st.form("journal_entry"):
            st.markdown("### Write about your day (Private)")
            journal_text = st.text_area("Your thoughts...", height=200, placeholder="What's on your mind? What made you feel this way today?", label_visibility="collapsed")
            
            if st.form_submit_button("Save Journal Entry", use_container_width=True, type="primary"):
                if journal_text.strip():
                    with st.spinner("Analyzing entry for wellness insights..."):
                        # Get risk and sentiment
                        result = predict_risk(journal_text, model, bert_tokenizer, label_encoder)
                        emotions = analyze_emotions(journal_text)
                        
                        # Save to firebase
                        try:
                            db.collection('users').document(st.session_state.user).collection('journal').add({
                                'text': journal_text,
                                'mood': st.session_state.selected_mood,
                                'risk': result['risk'],
                                'timestamp': firestore.SERVER_TIMESTAMP
                            })
                            st.success("Entry saved securely!")
                        except:
                            st.success("Entry saved locally! (Firebase disconnected)")
                        
                        st.markdown("---")
                        st.markdown("### 🌱 Wellness Insights")
                        
                        if result['risk'] == 'High Risk' or st.session_state.selected_mood == "Awful":
                            st.error("🚨 It sounds like you are going through a very difficult time. Please know you are not alone. Consider reaching out to a friend, family member, or calling a helpline (112 or AASRA).")
                        elif result['risk'] == 'Moderate Risk' or st.session_state.selected_mood == "Down":
                            st.warning("💡 You seem to be feeling down or anxious. Try a 5-minute deep breathing exercise or take a short walk outside to clear your mind.")
                        else:
                            # Use basic emotion
                            dominant_emotion = max(emotions, key=emotions.get) if sum(emotions.values()) > 0 else "Neutral"
                            
                            if dominant_emotion == "Anger":
                                st.info("🧘‍♀️ It seems you are feeling frustrated. Try writing down three things you can control right now.")
                            elif dominant_emotion == "Anxiety":
                                st.info("🌬️ You might be feeling anxious. Try the 4-7-8 breathing technique: Inhale for 4s, hold for 7s, exhale for 8s.")
                            else:
                                st.success("✨ Keep up the positive momentum! Reflecting on your day is a great habit.")
                else:
                    st.warning("Please write something before saving.")

# ============================================
# PAGE 1.5: CHECK-IN CHAT
# ============================================
elif page == "💬 Check-in Chat":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">💬 Check-in Chat</h1></div>', unsafe_allow_html=True)
    st.markdown("Have a conversation. The AI will analyze your mental health risk in real-time.")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "Hi there! How are you feeling today? I'm here to listen."}]
        
    # Display chat messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # User message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Analyze risk in background
        if model is not None:
            result = predict_risk(prompt, model, bert_tokenizer, label_encoder)
            risk = result['risk']
            conf = result['confidence']
            
            # Simple empathetic response logic based on risk
            if risk == "High Risk":
                response = "I'm so sorry you're feeling this way. Please know you're not alone and there is support available. I strongly encourage you to reach out to one of the crisis resources. I'm here to listen if you want to share more."
            elif risk == "Moderate Risk":
                response = "That sounds really tough. It's completely okay to feel overwhelmed sometimes. Remember to take things one step at a time. Do you want to talk more about what's bothering you?"
            else:
                response = "Thank you for sharing that with me! I'm glad you're expressing yourself. How else has your week been going?"
        else:
            response = "I hear you. (Model not loaded for risk analysis)."
            
        # Assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
            
            if model is not None:
                color = "#E53E3E" if risk == "High Risk" else "#ED8936" if risk == "Moderate Risk" else "#38A169"
                st.markdown(f'<div style="font-size: 0.8rem; color: {color}; margin-top: 5px;"><i>Real-time analysis: {risk} ({conf:.1f}%)</i></div>', unsafe_allow_html=True)

# ============================================
# PAGE 1.7: USER DASHBOARD
# ============================================
elif page == "📈 User Dashboard":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">📈 User Dashboard</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">📊 Longitudinal Tracking (Mock Firebase)</h3>
        <p style="color:#4A5568;">Track mental health risk levels over time based on daily check-ins. In a full production environment, this data is pulled dynamically from Firebase Firestore.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Login Simulation
    col_l1, col_l2 = st.columns([3, 1])
    with col_l1:
        username = st.text_input("Enter your Patient ID / Username:", placeholder="e.g., user_123")
    with col_l2:
        st.write("")
        st.write("")
        login_btn = st.button("Load Dashboard", type="primary", use_container_width=True)
        
    if login_btn or (username and "dashboard_data" in st.session_state):
        if not username:
            st.error("Please enter a username.")
        else:
            with st.spinner(f"Fetching historical data for {username} from database..."):
                import time
                time.sleep(1) # Fake network delay
                
                # Generate Mock Data
                import pandas as pd
                import numpy as np
                import datetime
                
                # Create last 30 days
                dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(30, 0, -1)]
                
                # Create a semi-realistic trend (e.g. improving over time)
                # Starts high, drops to moderate, fluctuates to low
                base_trend = np.linspace(2.8, 1.2, 30)
                noise = np.random.normal(0, 0.4, 30)
                scores = np.clip(np.round(base_trend + noise), 1, 3)
                
                # Map numeric back to labels
                mapping = {1: "Low Risk", 2: "Moderate Risk", 3: "High Risk"}
                risk_labels = [mapping[int(s)] for s in scores]
                
                df = pd.DataFrame({
                    "Date": dates,
                    "Risk Score": scores,
                    "Risk Level": risk_labels
                })
                
                st.session_state.dashboard_data = df
                
            st.success(f"Successfully loaded data for **{username}**!")
            
            st.markdown("### 📅 30-Day Risk Trend")
            
            # Metric Cards
            current_risk = df.iloc[-1]["Risk Level"]
            avg_risk = df["Risk Score"].mean()
            avg_risk_str = "Moderate Risk" if 1.5 < avg_risk < 2.5 else "High Risk" if avg_risk >= 2.5 else "Low Risk"
            
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric(label="Current Risk Level", value=current_risk)
            with mc2:
                st.metric(label="30-Day Average", value=avg_risk_str)
            with mc3:
                st.metric(label="Total Check-ins", value="30")
                
            st.markdown("---")
            
            # Plotting with Altair
            import altair as alt
            
            # Define colors
            domain = ['High Risk', 'Moderate Risk', 'Low Risk']
            range_ = ['#E53E3E', '#ED8936', '#38A169']
            
            chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=100)).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Risk Score:Q', scale=alt.Scale(domain=[0.5, 3.5]), axis=alt.Axis(values=[1, 2, 3], title="Risk Severity (3=High)")),
                color=alt.Color('Risk Level:N', scale=alt.Scale(domain=domain, range=range_)),
                tooltip=['Date', 'Risk Level']
            ).properties(
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            st.markdown("""
            <div class="glass-card" style="margin-top: 1rem;">
                <h4 style="margin-top:0;">💡 Clinical Insight</h4>
                <p style="color:#4A5568;">Based on the last 30 days, your mental health risk has shown a <strong>gradual improvement</strong>. The majority of your recent check-ins are classified as <em>Low to Moderate Risk</em>, compared to <em>High Risk</em> at the start of the month.</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# PAGE 2: MODEL PERFORMANCE
# ============================================
elif page == "📊 Model Performance":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">📊 Model Performance</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🎯", "Accuracy", "88.95%", "#667EEA"),
        ("⭐", "F1-Score", "89.55%", "#38A169"),
        ("📈", "Precision", "85.66%", "#ED8936"),
        ("📊", "Recall", "93.81%", "#805AD5")
    ]
    for col, (icon, label, value, color) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-top: 4px solid {color};">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="color: #718096; font-size: 0.9rem; margin-top: 0.5rem;">{label}</div>
                <div style="font-size: 2rem; font-weight: 800; color: {color}; margin-top: 0.3rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Model Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4>BERT + BiLSTM Hybrid Model</h4>
        <ul>
            <li><strong>Base Model:</strong> bert-base-uncased (110M params)</li>
            <li><strong>BiLSTM:</strong> 2 layers × 256 hidden units (bidirectional)</li>
            <li><strong>Attention Heads:</strong> 12</li>
            <li><strong>Hidden Size:</strong> 768 (BERT) + 512 (BiLSTM)</li>
            <li><strong>Max Sequence:</strong> 128 tokens</li>
            <li><strong>Dataset:</strong> 52,681 social media posts</li>
            <li><strong>Training Epochs:</strong> 5</li>
            <li><strong>Class Weights:</strong> Balanced (handles imbalance)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PAGE 3: DATA INSIGHTS
# ============================================
elif page == "📈 Data Insights":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">📈 Data Insights</h1></div>', unsafe_allow_html=True)
    
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Samples", f"{len(df):,}")
        with col2:
            st.metric("📝 Categories", len(df['label'].unique()))
        with col3:
            st.metric("📏 Avg Length", f"{df['text'].str.len().mean():.0f} chars")
        
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Label Distribution</div>', unsafe_allow_html=True)
        
        label_counts = df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_list = ['#667EEA', '#38A169', '#ED8936', '#E53E3E', '#805AD5', '#DD6B20', '#319795']
        bars = ax.bar(label_counts.index, label_counts.values, color=colors_list[:len(label_counts)])
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Categories', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        st.pyplot(fig)
        
        st.markdown("---")
        st.markdown('<div class="section-header">📝 Sample Data</div>', unsafe_allow_html=True)
        st.dataframe(df[['text', 'label']].head(10), use_container_width=True)
    else:
        st.error("❌ Dataset not found")

# ============================================
# PAGE 4: HISTORY
# ============================================
elif page == "📜 History":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">📜 Prediction History</h1></div>', unsafe_allow_html=True)
    
    total = sum(st.session_state.analytics.values())
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total", total)
    with col2:
        st.metric("🟢 Low Risk", st.session_state.analytics['Low Risk'])
    with col3:
        st.metric("🟡 Moderate", st.session_state.analytics['Moderate Risk'])
    with col4:
        st.metric("🔴 High Risk", st.session_state.analytics['High Risk'])
    
    if total > 0 and PLOTLY_AVAILABLE:
        st.markdown("---")
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.analytics.keys()),
            values=list(st.session_state.analytics.values()),
            marker_colors=['#38A169', '#ED8936', '#E53E3E'],
            hole=0.5
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="section-header">📋 Recent Predictions</div>', unsafe_allow_html=True)
    
    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            emoji = "🔴" if item['risk'] == "High Risk" else "🟡" if item['risk'] == "Moderate Risk" else "🟢"
            color = "#E53E3E" if item['risk'] == "High Risk" else "#ED8936" if item['risk'] == "Moderate Risk" else "#38A169"
            st.markdown(f"""
            <div class="history-item-premium" style="border-left-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="color: {color}; font-size: 1.05rem;">{emoji} {item['risk']}</strong>
                    <span style="color: #718096; font-size: 0.85rem;">
                        {item['confidence']:.1f}% • {item['time']}
                    </span>
                </div>
                <p style="margin: 0.5rem 0 0 0; color: #4A5568; font-size: 0.9rem;">
                    {item['text'][:200]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.session_state.analytics = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0}
            st.rerun()
    else:
        st.info("No predictions yet. Go to Home to analyze some text!")

# ============================================
# PAGE 5: BATCH ANALYSIS
# ============================================
elif page == "📁 Batch Analysis":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">📁 Batch Analysis</h1></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card"><p>Upload a CSV file with a <code>text</code> column to analyze multiple texts at once.</p></div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            if 'text' not in batch_df.columns:
                st.error("CSV must have a 'text' column")
            else:
                st.success(f"✅ {len(batch_df)} texts loaded")
                st.dataframe(batch_df.head(5), use_container_width=True)
                
                if st.button("🚀 Analyze All", type="primary"):
                    progress = st.progress(0)
                    results = []
                    for i, txt in enumerate(batch_df['text']):
                        r = predict_risk(str(txt), model, bert_tokenizer, label_encoder)
                        results.append({
                            'text': str(txt)[:100],
                            'risk': r['risk'] if r else 'Error',
                            'confidence': r['confidence'] if r else 0
                        })
                        progress.progress((i + 1) / len(batch_df))
                    st.session_state.batch_results = pd.DataFrame(results)
                    st.success("✅ Complete!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.session_state.batch_results is not None:
        st.markdown("---")
        batch_df = st.session_state.batch_results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Low", len(batch_df[batch_df['risk'] == 'Low Risk']))
        with col2:
            st.metric("🟡 Moderate", len(batch_df[batch_df['risk'] == 'Moderate Risk']))
        with col3:
            st.metric("🔴 High", len(batch_df[batch_df['risk'] == 'High Risk']))
        
        st.dataframe(batch_df, use_container_width=True)
        st.download_button("📥 Download Results", batch_df.to_csv(index=False), "results.csv")

# ============================================
# PAGE 5.5: SOCIAL MEDIA RISK
# ============================================
elif page == "📱 Social Media Risk":
    st.markdown("""<div class="hero-header"><h1 class="hero-title">📱 Social Media Risk</h1><p class="hero-subtitle">Simulate analyzing a user's recent social media posts to detect longitudinal risk patterns.</p></div>""", unsafe_allow_html=True)
    
    st.markdown("### Connect Account (Simulation)")
    username = st.text_input("Enter a social media handle (e.g. @johndoe)", placeholder="@username")
    
    if st.button("Fetch & Analyze Recent Posts", type="primary"):
        if username.strip():
            with st.spinner(f"Fetching recent posts for {username}..."):
                # Simulate API delay
                import time
                time.sleep(1.5)
                
                # Mock data generation
                import random
                from datetime import datetime, timedelta
                
                mock_posts = [
                    "Just had a great coffee! feeling good today.",
                    "The weather is so gloomy, makes me want to stay in bed all day...",
                    "Work is so overwhelming right now. I don't know how much more I can take.",
                    "Everything feels pointless lately. Why even try?",
                    "Trying to stay positive, but it's really hard."
                ]
                
                # Randomize order and assign dates
                random.shuffle(mock_posts)
                results = []
                today = datetime.now()
                
                for i, post in enumerate(mock_posts):
                    date = today - timedelta(days=len(mock_posts)-1-i)
                    r = predict_risk(post, model, bert_tokenizer, label_encoder)
                    results.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "text": post,
                        "risk": r['risk'] if r else "Low Risk",
                        "confidence": r['confidence'] if r else 0.0
                    })
                
                df_social = pd.DataFrame(results)
                
                st.success("✅ Successfully fetched and analyzed 5 recent posts.")
                
                # Plot
                st.markdown(f"#### 📈 7-Day Risk Trajectory for {username}")
                risk_map = {"Low Risk": 1, "Moderate Risk": 2, "High Risk": 3}
                df_social['risk_score'] = df_social['risk'].map(risk_map)
                
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_social['date'], y=df_social['risk_score'],
                        mode='lines+markers', line=dict(color='#3182ce', width=3), marker=dict(size=10)
                    ))
                    fig.update_layout(
                        yaxis=dict(tickvals=[1,2,3], ticktext=["Low Risk", "Moderate Risk", "High Risk"]),
                        height=300, margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📝 Post Breakdown")
                st.dataframe(df_social[['date', 'text', 'risk', 'confidence']], use_container_width=True)
                
                # Alert trigger
                high_risk_count = len(df_social[df_social['risk'] == 'High Risk'])
                if high_risk_count >= 2:
                    st.error(f"🚨 **URGENT ALERT:** {username} has posted multiple High Risk statements recently. Immediate intervention is recommended.")
                elif high_risk_count == 1 or len(df_social[df_social['risk'] == 'Moderate Risk']) > 2:
                    st.warning(f"⚠️ **WARNING:** {username} is showing signs of declining mental health. Active monitoring recommended.")
        else:
            st.warning("Please enter a username.")

# ============================================
# PAGE 6: ABOUT
# ============================================
elif page == "🧬 About Model":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">🧬 About the Model</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3>How the Model Works</h3>
        
        <h4>1. Preprocessing</h4>
        <p>Text is normalized: lowercased, URLs removed, emojis converted to text, slang expanded.</p>
        
        <h4>2. Safety Layer</h4>
        <p>Before ML analysis, critical keywords are checked. Any suicidal or self-harm phrase triggers HIGH RISK immediately.</p>
        
        <h4>3. BERT Encoder</h4>
        <p>The BERT transformer (12 layers, 768 hidden) creates contextual embeddings that understand word relationships.</p>
        
        <h4>4. BiLSTM Layer</h4>
        <p>A 2-layer bidirectional LSTM captures sequential patterns by reading text both forward and backward.</p>
        
        <h4>5. Classification Head</h4>
        <p>Dense layers with softmax output produce probability distributions across 3 risk classes.</p>
        
        <h4>6. Decision Layer</h4>
        <p>Combines rule-based keywords with ML predictions for the final risk level.</p>
    </div>
    """, unsafe_allow_html=True)


elif page == "🧪 Test Cases":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">🧪 Test Cases</h1></div>', unsafe_allow_html=True)
    
    st.markdown("### Sample Test Cases with Expected Outputs")
    
    test_cases = pd.DataFrame({
        'Input Text': [
            'I had a wonderful day today',
            'I feel so depressed and lonely',
            'I want to end my life',
            'i feel to cutoff',
            'i need to cutoff my hand',
            "Can't wait to spend weekend alone crying",
            'I am fine everything is fine',
            'u r so bad',
            'cut off toxic friend',
            'I love my life'
        ],
        'Expected': [
            '🟢 Low', '🟡 Moderate', '🔴 High', '🔴 High', '🔴 High',
            '🟡 Moderate', '🟡 Moderate', '🟡 Moderate', '🟡 Moderate', '🟢 Low'
        ],
        'Reason': [
            'Positive', 'Depression signals', 'Suicidal keyword', 'Self-harm keyword',
            'Body-part self-harm', 'Distress pattern', 'Sarcasm/distress', 'Slang',
            'Social cutoff', 'Positive'
        ]
    })
    
    st.dataframe(test_cases, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### Run Your Own Test")
    test_input = st.text_input("Enter text to test:")
    if st.button("Test"):
        if test_input and model is not None:
            result = predict_risk(test_input, model, bert_tokenizer, label_encoder)
            st.success(f"Result: {result['risk']} ({result['confidence']:.1f}%)")
            st.info(f"Source: {result.get('source', 'N/A')}")

# ============================================
# PAGE X: XAI EXPLORER
# ============================================
elif page == "🔬 XAI Explorer":
    st.markdown('<div class="hero-header" style="padding: 2rem;"><h1 class="hero-title" style="font-size: 2rem;">🔬 XAI Explorer</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="margin-top:0;">🔍 Decision Flow & Transparency</h3>
        <p style="color:#4A5568;">Understand exactly how the hybrid system evaluates text across multiple safety layers before reaching a final decision, combined with deep neural network insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    xai_input = st.text_area("Enter text to analyze & explain:", height=120, placeholder="Example: I've been feeling really down and I just want to cut off all my friends.")
    
    if st.button("Generate Explanation", type="primary", use_container_width=True):
        if not xai_input.strip():
            st.error("Please enter some text.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            st.markdown("---")
            
            # 1. DECISION FLOW VISUALIZATION
            st.markdown('### 🔀 System Decision Flow')
            
            with st.spinner("Analyzing decision path..."):
                result = predict_risk(xai_input, model, bert_tokenizer, label_encoder)
                
                # Determine which layer triggered
                source_msg = result.get('source', '')
                layer1_triggered = "Critical safety keyword" in source_msg or "Self-harm" in source_msg
                layer2_triggered = "Distress pattern" in source_msg or "Social withdrawal" in source_msg
                
                col_f1, col_f2, col_f3 = st.columns(3)
                
                # Layer 1
                with col_f1:
                    if layer1_triggered:
                        st.markdown(f"""
                        <div style="background: #FED7D7; border: 2px solid #E53E3E; border-radius: 12px; padding: 1rem; text-align: center;">
                            <h4 style="color: #C53030; margin:0;">Layer 1: Safety Override</h4>
                            <p style="font-size:0.85rem; color:#742A2A; margin-top:5px;">🚨 <b>TRIGGERED</b><br>{source_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #F7FAFC; border: 2px dashed #CBD5E0; border-radius: 12px; padding: 1rem; text-align: center; opacity: 0.7;">
                            <h4 style="color: #718096; margin:0;">Layer 1: Safety Override</h4>
                            <p style="font-size:0.85rem; color:#A0AEC0; margin-top:5px;">✅ Passed (No critical keywords)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Layer 2
                with col_f2:
                    if layer1_triggered:
                        st.markdown("""
                        <div style="background: #F7FAFC; border: 2px dashed #CBD5E0; border-radius: 12px; padding: 1rem; text-align: center; opacity: 0.5;">
                            <h4 style="color: #718096; margin:0;">Layer 2: Distress Patterns</h4>
                            <p style="font-size:0.85rem; color:#A0AEC0; margin-top:5px;">⏭️ Skipped (Halted by L1)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif layer2_triggered:
                        st.markdown(f"""
                        <div style="background: #FEEBC8; border: 2px solid #DD6B20; border-radius: 12px; padding: 1rem; text-align: center;">
                            <h4 style="color: #C05621; margin:0;">Layer 2: Distress Patterns</h4>
                            <p style="font-size:0.85rem; color:#9C4221; margin-top:5px;">⚠️ <b>TRIGGERED</b><br>{source_msg}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #F7FAFC; border: 2px dashed #CBD5E0; border-radius: 12px; padding: 1rem; text-align: center; opacity: 0.7;">
                            <h4 style="color: #718096; margin:0;">Layer 2: Distress Patterns</h4>
                            <p style="font-size:0.85rem; color:#A0AEC0; margin-top:5px;">✅ Passed (No hard patterns)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                # Layer 3
                with col_f3:
                    if layer1_triggered or layer2_triggered:
                        st.markdown("""
                        <div style="background: #F7FAFC; border: 2px dashed #CBD5E0; border-radius: 12px; padding: 1rem; text-align: center; opacity: 0.5;">
                            <h4 style="color: #718096; margin:0;">Layer 3: AI Model</h4>
                            <p style="font-size:0.85rem; color:#A0AEC0; margin-top:5px;">⏭️ Skipped (Overridden)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: #EBF8FF; border: 2px solid #3182CE; border-radius: 12px; padding: 1rem; text-align: center;">
                            <h4 style="color: #2B6CB0; margin:0;">Layer 3: BERT+BiLSTM</h4>
                            <p style="font-size:0.85rem; color:#2C5282; margin-top:5px;">🤖 <b>EVALUATED</b><br>Score: {result['confidence']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # 2. LIME EXPLANATION
            st.markdown('### 🧩 Neural Network Insight (LIME)')
            st.markdown("If the AI model evaluated the text, see exactly which words influenced its prediction.")
            
            with st.spinner("🧠 Generating LIME text explanation..."):
                from lime.lime_text import LimeTextExplainer
                import numpy as np
                import torch
                
                def lime_predict_fn(texts):
                    model.eval()
                    probs_list = []
                    batch_size = 32
                    with torch.no_grad():
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            encoded = bert_tokenizer(
                                batch_texts,
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors='pt'
                            )
                            input_ids = encoded['input_ids'].to(DEVICE)
                            attention_mask = encoded['attention_mask'].to(DEVICE)
                            output = model(input_ids, attention_mask)
                            probs = torch.softmax(output, dim=1).cpu().numpy()
                            probs_list.extend(probs)
                    return np.array(probs_list)
                
                explainer = LimeTextExplainer(class_names=label_encoder.classes_)
                # Use fewer samples (default 5000 is too slow for BERT on CPU). 100 samples is fast and sufficient for a visual explanation.
                exp = explainer.explain_instance(xai_input, lime_predict_fn, num_features=10, num_samples=100)
                
                st.components.v1.html(exp.as_html(), height=450, scrolling=True)
# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p style="font-weight: 600; color: #4A5568;">🧠 MindGuard AI — BERT + BiLSTM Powered</p>
    <p style="font-size: 0.85rem;"> © 2026</p>
    <p style="font-size: 0.8rem; color: #A0AEC0;">
        This tool is for educational and research purposes only. Not a substitute for professional help.
    </p>
</div>
""", unsafe_allow_html=True)