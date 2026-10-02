"""
Complete Working Depression Detection Model
For Mental Health Text Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

print("="*70)
print("🧠 AI-Driven Depression Detection & Mental Health Risk Classification")
print("📝 Using Social Media Text Data")
print("="*70)

# ============================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================
print("\n📂 Loading and cleaning dataset...")

# Load data
df = pd.read_csv('data.csv')

# Convert all text to string and handle NaN
df['text'] = df['text'].astype(str).fillna('')
df = df[df['text'].str.strip() != '']
df = df.dropna(subset=['text'])

print(f"✅ Loaded {len(df)} samples")
print(f"📊 Columns: {df.columns.tolist()}")

# Show label distribution
print(f"\n📊 Label distribution:")
print(df['label'].value_counts())

# ============================================
# STEP 2: MAP LABELS TO RISK LEVELS
# ============================================
print("\n🔄 Mapping labels to risk levels...")

# Map mental health conditions to risk levels
risk_mapping = {
    'Normal': 'Low',
    'Depression': 'Moderate',
    'Anxiety': 'Moderate',
    'Stress': 'Moderate',
    'Bipolar': 'High',
    'Suicidal': 'High',
    'Personality disorder': 'High'
}

# Check if all labels are in mapping
unique_labels = df['label'].unique()
print(f"📊 Unique labels found: {unique_labels}")

# Apply mapping
df['risk_level'] = df['label'].map(risk_mapping)

# Handle any unmapped labels
if df['risk_level'].isna().any():
    print(f"⚠️ Some labels not in mapping. Using original labels.")
    df['risk_level'] = df['label']

df = df.dropna(subset=['risk_level'])
print(f"\n📊 Risk level distribution:")
print(df['risk_level'].value_counts())

# ============================================
# STEP 3: PREPROCESS TEXT
# ============================================
print("\n🔄 Preprocessing text...")

# Get stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text"""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty texts after cleaning
df = df[df['cleaned_text'].str.strip() != '']
print(f"✅ After cleaning: {len(df)} samples")

# ============================================
# STEP 4: ENCODE LABELS
# ============================================
print("\n📊 Encoding labels...")

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['risk_level'])

print(f"📊 Classes: {le.classes_}")

# ============================================
# STEP 5: TOKENIZE AND PREPARE FOR LSTM
# ============================================
print("\n🔄 Tokenizing text...")

# Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])

# Convert to sequences
sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
max_len = 128
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = df['label_encoded'].values

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")
print(f"✅ Number of classes: {len(le.classes_)}")

# ============================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================
print("\n📊 Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training: {len(X_train)} samples")
print(f"✅ Test: {len(X_test)} samples")

# ============================================
# STEP 7: BUILD LSTM MODEL
# ============================================
print("\n🧠 Building LSTM Model...")

model = Sequential([
    Embedding(10000, 128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
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
print("\n🚀 Training Model...")

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True)
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# STEP 9: EVALUATE
# ============================================
print("\n📊 Evaluating Model...")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)
print(f"✅ Accuracy:  {accuracy*100:.2f}%")
print(f"✅ Precision: {precision*100:.2f}%")
print(f"✅ Recall:    {recall*100:.2f}%")
print(f"✅ F1-Score:  {f1*100:.2f}%")
print("="*70)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# ============================================
# STEP 10: CONFUSION MATRIX
# ============================================
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix - Mental Health Risk Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ============================================
# STEP 11: TRAINING HISTORY
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training history saved as 'training_history.png'")
plt.show()

# ============================================
# STEP 12: SAVE MODEL
# ============================================
model.save('depression_model.h5')
print("\n✅ Model saved as 'depression_model.h5'")

print("\n" + "="*70)
print("✅ Training Complete!")
print("📊 Model Performance:")
print(f"   - Accuracy:  {accuracy*100:.2f}%")
print(f"   - Precision: {precision*100:.2f}%")
print(f"   - Recall:    {recall*100:.2f}%")
print(f"   - F1-Score:  {f1*100:.2f}%")
print("="*70)