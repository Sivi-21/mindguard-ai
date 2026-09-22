"""
BERT + BiLSTM Hybrid Training
Target: Match paper metrics (88.95% accuracy, 93.81% recall)
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# CONFIG
# ============================================
DATA_PATH = "data.csv"
BERT_MODEL = "bert-base-uncased"
OUTPUT_MODEL = "mental_health_bert_bilstm_model"
LABEL_ENCODER_PATH = "label_encoder_bilstm.pkl"

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# DATASET
# ============================================
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ============================================
# HYBRID MODEL: BERT + BiLSTM
# ============================================
class BertBiLSTM(nn.Module):
    def __init__(self, bert_model_name, num_classes=3, hidden_size=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size  # 768 for bert-base

        # BiLSTM layers on top of BERT
        self.bilstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # ×2 for bidirectional

    def forward(self, input_ids, attention_mask):
        # BERT output: (batch, seq_len, 768)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state

        # BiLSTM: (batch, seq_len, hidden*2)
        lstm_out, _ = self.bilstm(sequence_output)

        # Use CLS token position (first token) from LSTM output
        cls_repr = lstm_out[:, 0, :]

        # Classify
        x = self.dropout(cls_repr)
        logits = self.classifier(x)
        return logits

# ============================================
# LOAD DATA
# ============================================
print("\n📂 Loading data...")
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

# Map labels
label_mapping = {
    "Normal": "Low", "Depression": "Moderate", "Anxiety": "Moderate",
    "Stress": "Moderate", "Suicidal": "High", "Bipolar": "High",
    "Personality disorder": "High",
}
df["risk"] = df["label"].map(label_mapping)
df = df.dropna(subset=["risk"])

print(f"✅ Loaded {len(df)} samples")
print(df["risk"].value_counts())

# Encode labels
le = LabelEncoder()
df["risk_id"] = le.fit_transform(df["risk"])
print(f"\n📊 Classes: {list(le.classes_)}")

with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(le, f)

# ============================================
# SPLIT
# ============================================
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["risk_id"], random_state=42
)
print(f"✅ Train: {len(train_df)} | Val: {len(val_df)}")

# ============================================
# CLASS WEIGHTS
# ============================================
class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(df["risk_id"].values),
    y=df["risk_id"].values,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"\n⚖️  Class weights: {dict(zip(le.classes_, class_weights.round(3)))}")

# ============================================
# TOKENIZER & DATASET
# ============================================
print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

train_dataset = MentalHealthDataset(train_df["text"].tolist(), train_df["risk_id"].tolist(), tokenizer, MAX_LEN)
val_dataset = MentalHealthDataset(val_df["text"].tolist(), val_df["risk_id"].tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================
# MODEL
# ============================================
print("\n📥 Building BERT + BiLSTM model...")
model = BertBiLSTM(BERT_MODEL, num_classes=len(le.classes_))
model.to(DEVICE)

# ============================================
# OPTIMIZER
# ============================================
# Different learning rates: BERT smaller, BiLSTM bigger
optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": LR},
    {"params": model.bilstm.parameters(), "lr": LR * 5},
    {"params": model.classifier.parameters(), "lr": LR * 5},
])

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

# ============================================
# TRAIN
# ============================================
print("\n" + "=" * 60)
print("🎯 Training BERT + BiLSTM")
print("=" * 60)

best_f1 = 0.0
best_recall = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"   Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print(f"\n📊 Epoch {epoch}/{EPOCHS}")
    print(f"   Loss:      {avg_loss:.4f}")
    print(f"   Accuracy:  {acc*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1:        {f1*100:.2f}%")

    # Save best
    if f1 > best_f1:
        best_f1 = f1
        best_recall = recall
        os.makedirs(OUTPUT_MODEL, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(OUTPUT_MODEL, "model.pt"))
        tokenizer.save_pretrained(OUTPUT_MODEL)
        print(f"   ⭐ New best F1! Saved.")

print("\n" + "=" * 60)
print(f"✅ Training Complete!")
print(f"   Best F1: {best_f1*100:.2f}%")
print(f"   Best Recall: {best_recall*100:.2f}%")
print("=" * 60)

# Final evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].tolist())

print("\n📊 FINAL CLASSIFICATION REPORT:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# Save label encoder copy in the model folder too
import shutil
shutil.copy(LABEL_ENCODER_PATH, os.path.join(OUTPUT_MODEL, "label_encoder.pkl"))
print(f"\n✅ Model saved to: {OUTPUT_MODEL}/")
print(f"✅ Label encoder saved to: {OUTPUT_MODEL}/label_encoder.pkl")