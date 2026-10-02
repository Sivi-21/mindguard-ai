"""
Improved Depression Detection Model Training
- Handles class imbalance with class weights
- Uses better preprocessing
- Uses more epochs with early stopping
- Saves tokenizer + label encoder
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import re
import os
import pickle

print("="*70)
print("🧠 Training IMPROVED Depression Detection Model")
print("="*70)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n📂 Loading data...")

df = pd.read_csv('data.csv')
df['text'] = df['text'].astype(str).fillna('')
df = df[df['text'].str.strip() != '']

print(f"✅ Loaded {len(df)} samples")
print(f"📊 Original label distribution:")
print(df['label'].value_counts())

# ============================================
# STEP 2: MAP LABELS TO 3 RISK LEVELS
# ============================================
print("\n🔄 Mapping to 3 risk levels...")

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
# STEP 3: ENHANCED PREPROCESSING
# ============================================
print("\n🔄 Enhanced preprocessing...")

# Slang dictionary
slang_map = {
    'u': 'you', 'ur': 'your', 'r': 'are', 'y': 'why',
    'gr8': 'great', 'lol': 'laughing out loud', 'lmao': 'laughing my ass off',
    'idk': 'i do not know', 'imo': 'in my opinion', 'btw': 'by the way',
    'smh': 'shaking my head', 'gonna': 'going to', 'wanna': 'want to',
    'gotta': 'got to', 'kinda': 'kind of', 'nah': 'no', 'yeah': 'yes',
    'plz': 'please', 'thx': 'thanks', 'ty': 'thank you',
    'cant': 'cannot', "can't": 'cannot', 'dont': 'do not', "don't": 'do not',
    'wont': 'will not', "won't": 'will not', 'isnt': 'is not', "isn't": 'is not',
    'wasnt': 'was not', "wasn't": 'was not', 'didnt': 'did not', "didn't": 'did not',
    'couldnt': 'could not', "couldn't": 'could not', 'shouldnt': 'should not',
    'wouldnt': 'would not', 'im': 'i am', "i'm": 'i am', 'ive': 'i have',
    'youre': 'you are', "you're": 'you are', 'theyre': 'they are',
    'its': 'it is', "it's": 'it is', 'thats': 'that is', 'whats': 'what is'
}

def clean_text(text):
    text = str(text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Expand slang & contractions (word-level)
    words = text.split()
    words = [slang_map.get(w, w) for w in words]
    text = ' '.join(words)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty
df = df[df['cleaned_text'].str.strip() != '']
print(f"✅ After cleaning: {len(df)} samples")

# ============================================
# STEP 4: ENCODE LABELS
# ============================================
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['risk_level'])

print(f"\n📊 Classes: {list(le.classes_)}")
print(f"📊 Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ============================================
# STEP 5: COMPUTE CLASS WEIGHTS (Fix Imbalance!)
# ============================================
print("\n⚖️ Computing class weights to handle imbalance...")

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

MAX_WORDS = 20000
MAX_LEN = 128

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as 'tokenizer.pkl'")

sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label_encoded'].values

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

# ============================================
# STEP 7: SPLIT DATA
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

# ============================================
# STEP 8: BUILD IMPROVED MODEL
# ============================================
print("\n🧠 Building IMPROVED model...")

model = Sequential([
    # Embedding layer
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    SpatialDropout1D(0.3),
    
    # Bidirectional LSTM layers
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    
    # Dense layers
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    # Output
    Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 9: TRAIN
# ============================================
print("\n🚀 Training model...")

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=25,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 10: EVALUATE
# ============================================
print("\n📊 Evaluating...")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

# Detailed report
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# ============================================
# STEP 11: SAVE MODEL
# ============================================
model.save('depression_model.h5')
print("\n✅ Model saved as 'depression_model.h5'")

size = os.path.getsize('depression_model.h5') / 1024 / 1024
print(f"📊 Model size: {size:.2f} MB")

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Label encoder saved as 'label_encoder.pkl'")

# ============================================
# STEP 12: QUICK TEST
# ============================================
print("\n" + "="*70)
print("🧪 QUICK TEST WITH NEW MODEL")
print("="*70)

def test_predict(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded, verbose=0)[0]
    labels = list(le.classes_)
    idx = np.argmax(pred)
    return labels[idx], pred[idx] * 100, pred

test_texts = [
    "I had a wonderful day today",
    "I feel so happy and grateful",
    "I feel so depressed and lonely",
    "I want to end my life",
    "severe attack",
    "I'm feeling anxious",
    "u r so bad",
    "I'm so stressed out",
    "Nothing matters anymore",
    "help me please"
]

for text in test_texts:
    label, conf, probs = test_predict(text)
    probs_str = " | ".join([f"{l}: {p*100:.1f}%" for l, p in zip(le.classes_, probs)])
    print(f"\n📝 '{text}'")
    print(f"   → {probs_str}")
    print(f"   🎯 {label} ({conf:.1f}%)")

print("\n" + "="*70)
print("✅ Training Complete!")
print("="*70)