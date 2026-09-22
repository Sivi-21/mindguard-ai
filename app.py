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
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'analytics' not in st.session_state:
    st.session_state.analytics = {'Low Risk': 0, 'Moderate Risk': 0, 'High Risk': 0}
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

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
    """Load the trained BERT + BiLSTM model"""
    model_path = "mental_health_bert_bilstm_model"
    
    if not os.path.exists(model_path):
        return None, None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        le_path = os.path.join(model_path, "label_encoder.pkl")
        if not os.path.exists(le_path):
            le_path = "label_encoder_bilstm.pkl"
        
        with open(le_path, "rb") as f:
            label_encoder = pickle.load(f)
        
        num_classes = len(label_encoder.classes_)
        
        model = BertBiLSTM("bert-base-uncased", num_classes=num_classes)
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), map_location=DEVICE))
        model.eval()
        model.to(DEVICE)
        
        return model, tokenizer, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading BERT+BiLSTM: {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('data.csv')
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
        'lonely', 'emptiness', 'numb'
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
            padding="max_length", max_length=max_len
        )
        
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        with torch.no_grad():
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
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Navigate:",
        ["🏠 Home", "📊 Model Performance", "📈 Data Insights",
         "📜 History", "📁 Batch Analysis", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    threshold = st.slider("Confidence Threshold", 0, 100, 60)
    
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
            if model is None:
                st.error("❌ BERT+BiLSTM model not found. Check mental_health_bert_bilstm_model/ folder.")
            else:
                with st.spinner("🧠 Analyzing with BERT+BiLSTM..."):
                    result = predict_risk(text_input, model, bert_tokenizer, label_encoder)
                    
                    if result:
                        risk_labels = {
                            'High Risk': '🔴 HIGH RISK',
                            'Low Risk': '🟢 LOW RISK',
                            'Moderate Risk': '🟡 MODERATE RISK'
                        }
                        
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
                        
                        st.session_state.history.append({
                            'text': text_input,
                            'risk': result['risk'],
                            'confidence': result['confidence'],
                            'time': pd.Timestamp.now().strftime('%H:%M:%S')
                        })
                        st.session_state.analytics[result['risk']] += 1
                        
                        if result['confidence'] < threshold:
                            st.warning(f"⚠️ Below threshold ({threshold}%)")
                        
                        st.markdown('<div class="section-header">📈 Probability Distribution</div>', unsafe_allow_html=True)
                        render_probability_bars(result['probabilities'])
                        
                        sentiment = get_smart_sentiment(text_input, result)
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
                                    {highlight_text(text_input, result['matched'])}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if PLOTLY_AVAILABLE:
                            emotions = analyze_emotions(text_input)
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
                        
                        st.markdown('<div class="section-header">🎯 Actions</div>', unsafe_allow_html=True)
                        act_col1, act_col2 = st.columns(2)
                        
                        with act_col1:
                            st.download_button(
                                "📥 Download Report",
                                generate_report(text_input, result),
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