"""
Train Depression Detection Model
Properly trains and saves the model with real weights
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re
import os
import pickle

print("="*60)
print("🧠 Training Depression Detection Model")
print("="*60)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n📂 Loading data...")
df = pd.read_csv('data.csv')
df['text'] = df['text'].astype(str).fillna('')
df = df[df['text'].str.strip() != '']

print(f"✅ Loaded {len(df)} samples")

# ============================================
# STEP 2: MAP LABELS TO RISK LEVELS
# ============================================
print("\n🔄 Mapping labels...")

risk_mapping = {
    'Normal': 'Low',
    'Depression': 'Moderate',
    'Anxiety': 'Moderate',
    'Stress': 'Moderate',
    'Bipolar': 'High',
    'Suicidal': 'High',
    'Personality disorder': 'High'
}

df['risk_level'] = df['label'].map(risk_mapping)
df = df.dropna(subset=['risk_level'])

print(f"📊 Risk level distribution:")
print(df['risk_level'].value_counts())

# ============================================
# STEP 3: PREPROCESS TEXT
# ============================================
print("\n🔄 Preprocessing text...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# ============================================
# STEP 4: ENCODE LABELS
# ============================================
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['risk_level'])
print(f"📊 Classes: {le.classes_}")

# ============================================
# STEP 5: TOKENIZE
# ============================================
print("\n🔄 Tokenizing...")

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as tokenizer.pkl")

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
max_len = 128
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['label_encoded'].values

# ============================================
# STEP 6: SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# ============================================
# STEP 7: BUILD MODEL
# ============================================
print("\n🧠 Building model...")

model = Sequential([
    Embedding(10000, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 8: TRAIN MODEL
# ============================================
print("\n🚀 Training model...")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=15,
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 9: EVALUATE
# ============================================
print("\n📊 Evaluating...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# ============================================
# STEP 10: SAVE MODEL
# ============================================
model.save('depression_model.h5')
print("\n✅ Model saved as depression_model.h5")

size = os.path.getsize('depression_model.h5')
print(f"📊 Model size: {size / 1024 / 1024:.2f} MB")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Label encoder saved as label_encoder.pkl")

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)