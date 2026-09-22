"""
Improved Depression Detection Model with Class Weights
Handles class imbalance and improves predictions
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import re
import os
import pickle

print("="*60)
print("🧠 Training Improved Depression Detection Model")
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
# STEP 2: MAP LABELS
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
# STEP 5: COMPUTE CLASS WEIGHTS
# ============================================
print("\n⚖️ Computing class weights...")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(df['label_encoded']),
    y=df['label_encoded']
)
class_weight_dict = dict(enumerate(class_weights))
print(f"📊 Class weights: {class_weight_dict}")

# ============================================
# STEP 6: TOKENIZE
# ============================================
print("\n🔄 Tokenizing...")

tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
max_len = 128
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['label_encoded'].values

# ============================================
# STEP 7: SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# ============================================
# STEP 8: BUILD IMPROVED MODEL
# ============================================
print("\n🧠 Building improved model...")

model = Sequential([
    Embedding(20000, 128, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 9: TRAIN MODEL
# ============================================
print("\n🚀 Training improved model...")

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    class_weight=class_weight_dict,  # Handle imbalance
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 10: EVALUATE
# ============================================
print("\n📊 Evaluating...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# ============================================
# STEP 11: SAVE MODEL
# ============================================
model.save('depression_model.h5')
print("\n✅ Model saved as depression_model.h5")

size = os.path.getsize('depression_model.h5')
print(f"📊 Model size: {size / 1024 / 1024:.2f} MB")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)