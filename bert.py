import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Load your dataset
df = pd.read_csv('data.csv')  # replace with your path

# 2. Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['status'])

# 3. Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['statement'], df['label_encoded'], test_size=0.3, random_state=42
)
train_texts = train_texts.astype(str).tolist()
val_texts = val_texts.astype(str).tolist()

# 4. Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# 5. Create dataset
class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = MentalHealthDataset(train_encodings, train_labels.tolist())
val_dataset = MentalHealthDataset(val_encodings, val_labels.tolist())

# 6. Create model
num_labels = len(label_encoder.classes_)
print(label_encoder.classes_)
print(f"Number of labels: {num_labels}")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="epoch",  # <-- Evaluate every epoch
    logging_dir='./logs',
    logging_steps=10,
    report_to="wandb",  # <-- Enable wandb
    logging_first_step=True,
    save_strategy="epoch",  # Optional: Save model every epoch
    load_best_model_at_end=True  # Optional: Useful for later use
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 7. Evaluate the model
eval_result = trainer.evaluate()
print(eval_result)

# 8. Get predictions on the validation set
predictions_output = trainer.predict(val_dataset)
preds = predictions_output.predictions.argmax(-1)  # Get predicted class indices
labels = predictions_output.label_ids              # True labels

# 9. Print or save predictions and true labels
df_preds = pd.DataFrame({
    'true_label': labels,
    'predicted_label': preds
})
print(df_preds.head())

# 10. Optionally, save to CSV for further analysis
df_preds.to_csv('preds/val_predictions_bert.csv', index=False)

# 11. Save model
model.save_pretrained('results/BERT')

# *** To load the model later:
# loaded_model = BertForSequenceClassification.from_pretrained('results/BERT')
# loaded_model.eval()