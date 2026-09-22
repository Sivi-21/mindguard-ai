import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import os

# 1. Load your dataset
df = pd.read_csv('data.csv')  # Replace with your CSV path

# 2. Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['status'])

# 3. Tokenize text
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# 4. Build vocabulary
counter = Counter()
for text in df['statement']:
    tokens = tokenize(str(text))
    counter.update(tokens)

vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text):
    return [vocab.get(word, vocab["<UNK>"]) for word in tokenize(text)]

# 5. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['statement'].astype(str), df['label_encoded'], test_size=0.3, random_state=42
)

# 6. Dataset class
class LSTMDataset(Dataset):
    def __init__(self, texts, labels):
        self.sequences = [torch.tensor(encode(text)) for text in texts]
        self.labels = torch.tensor(labels.tolist())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# 7. Collate function for padding
def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels)

# 8. Dataloaders
train_dataset = LSTMDataset(train_texts, train_labels)
val_dataset = LSTMDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

# 9. Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

# 10. Initialize model
num_labels = len(label_encoder.classes_)
model = LSTMClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 11. Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 12. Training loop with validation
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = [x.to(device) for x in batch]
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss:.4f} | Val Accuracy: {val_acc:.4f}")

# 13. Save validation predictions
df_preds = pd.DataFrame({
    'true_label': all_labels,
    'predicted_label': all_preds
})
df_preds.to_csv('preds/val_predictions_lstm.csv', index=False)
print(df_preds.head())

# 14. Save model
os.makedirs('results/LSTM', exist_ok=True)
torch.save(model.state_dict(), 'results/LSTM/lstm_model.pt')

# *** To load the model later:
# model = LSTMClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, output_dim=num_labels)
# model.load_state_dict(torch.load('results/LSTM/lstm_model.pt'))