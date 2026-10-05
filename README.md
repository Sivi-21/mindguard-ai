# 🧠 MindGuard AI - Phase 2

MindGuard AI is a comprehensive, advanced Mental Health Risk Assessment platform powered by a hybrid **BERT + BiLSTM** deep learning model. The platform is designed to provide real-time, accurate, and explainable mental health analysis for individuals and clinical professionals.

## 🚀 Key Features

### 1. 🧠 Core ML Architecture
- **Hybrid Model**: Utilizes a `bert-base-uncased` transformer combined with a 2-layer Bidirectional LSTM for deep contextual and sequential understanding of text.
- **Explainable AI (XAI)**: Integrated with LIME (Local Interpretable Model-agnostic Explanations) so users can see exactly *which words* contributed most heavily to the model's prediction.
- **Continuous Learning Loop**: Users can confirm or correct model predictions, sending feedback directly to a secure Firestore database to continuously retrain and improve the model over time.

### 2. 🩺 Clinical & Professional Tools
- **🧑‍⚕️ Therapist Dashboard**: A secure portal for clinicians to select patients and visualize their risk trajectory over time via interactive Plotly line charts.
- **📱 Social Media Risk**: Simulates longitudinal risk analysis by fetching recent social media posts, analyzing them in batch, and triggering urgent clinical warnings if a user's mental health state is actively declining.
- **📁 Batch Analysis**: Upload a CSV of texts to analyze hundreds of statements in seconds.
- **📥 PDF Reports**: Generate and download beautifully formatted, formal PDF reports summarizing risk and sentiment for clinical records.

### 3. 🧘‍♀️ User & Wellness Features
- **📓 Interactive Mood Journal**: A private diary where users can log their mood using emojis and write entries. The app dynamically analyzes the text and provides contextual wellness insights (e.g., suggesting a 4-7-8 breathing exercise if anxiety is detected).
- **🎙️ Voice Analysis**: Speak directly into your microphone or upload an audio file (`WAV`/`MP3`). The app uses `pydub` and Google Speech Recognition to instantly transcribe your speech and run risk analysis.
- **💬 Check-in Chat**: A conversational chatbot interface that analyzes your risk in real-time as you chat.
- **📍 Geolocation Resource Mapping**: When High Risk is detected, the app pings the user's IP to find their city and renders an interactive Folium map showing nearby (simulated) mental health clinics.
- **📧 Automated Email Alerts**: Employs an SMTP background thread to instantly email an Admin/Therapist if a severe High-Risk case is detected.

## 🛠️ Tech Stack
- **Frontend**: Streamlit, Streamlit-Folium
- **Backend / ML**: PyTorch, Transformers (HuggingFace), Scikit-Learn
- **NLP / Explanability**: LIME, TextBlob, SpeechRecognition, pydub
- **Database**: Google Firebase (Firestore)
- **Data Viz**: Plotly, Matplotlib, Folium

## 🏃‍♂️ How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install streamlit torch transformers scikit-learn textblob firebase-admin plotly fpdf2 folium streamlit-folium SpeechRecognition pydub
   ```

2. **Run the App**
   ```bash
   python -m streamlit run app.py
   ```

3. **Email Alerts (Optional)**
   - Go to `Settings -> Configure Email Alerts`.
   - Enter your Gmail and an [App Password](https://support.google.com/accounts/answer/185833?hl=en).

## ⚠️ Disclaimer
MindGuard AI is an experimental AI screening tool designed for demonstration purposes. It is **NOT** a medical diagnosis tool. If you or someone you know is in crisis, please contact emergency services immediately (e.g., 112 or local helplines).