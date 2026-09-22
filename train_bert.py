"""
train_bert.py
Fine-tunes mental/mental-bert-base-uncased on data.csv
Maps labels to 3 risk levels: Low, Moderate, High
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# CONFIGURATION
# ============================================
DATA_PATH = Path("data.csv")
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = Path("./mental_health_bert_model")
LABEL_ENCODER_PATH = Path("label_encoder.pkl")

BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Using device: {DEVICE}")


# ============================================
# DATASET CLASS
# ============================================
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================
# MAIN TRAINING
# ============================================
def main():
    print("=" * 60)
    print("🚀 Training BERT for Mental Health Detection")
    print("=" * 60)

    # ============================================
    # 1. Load data
    # ============================================
    print("\n📂 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} samples")
    print(f"📋 Columns: {df.columns.tolist()}")

    # Detect column names
    text_col = None
    label_col = None
    for col in df.columns:
        if col.lower() in ["text", "statement", "post", "content", "tweet"]:
            text_col = col
        if col.lower() in ["label", "status", "class", "category", "sentiment"]:
            label_col = col

    if text_col is None or label_col is None:
        print(f"❌ Could not detect text/label columns")
        return

    print(f"✅ Text column: '{text_col}'")
    print(f"✅ Label column: '{label_col}'")

    # ============================================
    # 2. Map labels to 3 risk levels
    # ============================================
    print("\n🔄 Mapping labels to risk levels...")

    label_mapping = {
        "Normal": "Low",
        "Depression": "Moderate",
        "Anxiety": "Moderate",
        "Stress": "Moderate",
        "Suicidal": "High",
        "Bipolar": "High",
        "Personality disorder": "High",
    }

    unique_labels = df[label_col].unique()
    print(f"📊 Unique labels in dataset: {unique_labels}")

    df["risk"] = df[label_col].map(label_mapping)

    unmapped = df[df["risk"].isna()][label_col].unique()
    if len(unmapped) > 0:
        print(f"⚠️ Unmapped labels (will be dropped): {unmapped}")

    df = df.dropna(subset=["risk", text_col])
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""]

    print(f"✅ After mapping: {len(df)} samples")
    print(f"📊 Risk distribution:")
    print(df["risk"].value_counts())

    # ============================================
    # 3. Encode labels
    # ============================================
    le = LabelEncoder()
    df["risk_id"] = le.fit_transform(df["risk"])
    print(f"\n📊 Classes: {list(le.classes_)}")
    print(f"📊 Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"✅ Label encoder saved to {LABEL_ENCODER_PATH}")

    # ============================================
    # 4. Compute class weights (FIXED)
    # ============================================
    print("\n⚖️  Computing class weights...")

    unique_classes = np.unique(df["risk_id"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=df["risk_id"].values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    for cls_id, w in zip(unique_classes, class_weights):
        cls_name = le.inverse_transform([cls_id])[0]
        print(f"   {cls_name}: {w:.3f}")

    # ============================================
    # 5. Train/validation split
    # ============================================
    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["risk_id"],
        random_state=42,
    )

    print(f"\n✅ Train: {len(train_df)} samples")
    print(f"✅ Val:   {len(val_df)} samples")

    # ============================================
    # 6. Tokenizer & Datasets
    # ============================================
    print("\n📥 Loading BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MentalHealthDataset(
        train_df[text_col].tolist(),
        train_df["risk_id"].tolist(),
        tokenizer,
    )
    val_dataset = MentalHealthDataset(
        val_df[text_col].tolist(),
        val_df["risk_id"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ============================================
    # 7. Model, optimizer, loss
    # ============================================
    print("📥 Loading BERT model...")
    print("   ⏳ This will download ~400MB on first run")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    )
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    # ============================================
    # 8. Training loop
    # ============================================
    print("\n" + "=" * 60)
    print("🎯 Starting Training")
    print("=" * 60)

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ----- Training -----
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 100 == 0:
                print(f"   Epoch {epoch} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # ----- Validation -----
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        print("\n" + "-" * 60)
        print(f"📊 EPOCH {epoch}/{EPOCHS} RESULTS")
        print("-" * 60)
        print(f"   Train Loss:  {avg_loss:.4f}")
        print(f"   Val Acc:     {acc:.4f}")
        print(f"   Precision:   {precision:.4f}")
        print(f"   Recall:      {recall:.4f}")
        print(f"   F1 Score:    {f1:.4f}")
        print("-" * 60)

        if f1 > best_f1:
            best_f1 = f1
            print(f"   ⭐ New best F1! Saving...")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    # ============================================
    # 9. Final Save
    # ============================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Model saved to {OUTPUT_DIR.resolve()}")
    print(f"✅ Best F1: {best_f1:.4f}")

    # ============================================
    # 10. Demo predictions
    # ============================================
    print("\n" + "=" * 60)
    print("🧪 Demo Predictions")
    print("=" * 60)

    demo_sentences = [
        "I had a wonderful day today",
        "I feel so depressed and lonely",
        "I want to end my life",
        "severe attack",
        "i need to cutoff my hand",
        "u r so bad",
        "I'm feeling anxious",
        "nothing matters anymore",
        "help me please",
        "I love my life",
    ]

    model.eval()
    for sent in demo_sentences:
        enc = tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()
            pred_label = le.inverse_transform([pred_id])[0]
            confidence = probs[pred_id].item() * 100

        emoji = "🟢" if pred_label == "Low" else "🟡" if pred_label == "Moderate" else "🔴"
        print(f"\n📝 '{sent}'")
        print(f"   {emoji} {pred_label} ({confidence:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()