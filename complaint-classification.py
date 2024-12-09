import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import pandas as pd
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Dataset class
class ComplaintDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.input_ids = []
        self.attention_masks = []
        self.labels = data["Issue"]

        for text in data["Processed Narrative"]:
            encoded = tokenizer.encode_plus(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids.append(encoded["input_ids"].squeeze(0))
            self.attention_masks.append(encoded["attention_mask"].squeeze(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = {
            "input_ids": self.input_ids[idx].detach().clone(),
            "attention_mask": self.attention_masks[idx].detach().clone(),
            "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long).detach().clone(),
        }
        return data


# Load and preprocess data
df = pd.read_csv("/home/ubuntu/NLP/final_df.csv")
print(f"Original dataset size: {len(df)}")
df = df[df["Issue"].map(df["Issue"].value_counts()) > 5]  # Filter rare classes
print(f"Filtered dataset size: {len(df)}")
df["Issue"] = df["Issue"].astype("category").cat.codes

# Compute class weights
class_weights = torch.tensor(1 / df["Issue"].value_counts(normalize=True).values, dtype=torch.float32)

# Verify target names
target_names = list(df["Issue"].astype("category").cat.categories.astype(str))
print("Target names:", target_names)

# Split data
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Issue"], random_state=42)

# Tokenizer and Dataset
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataset = ComplaintDataset(train_data, tokenizer, max_len=128)
test_dataset = ComplaintDataset(test_data, tokenizer, max_len=128)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(df["Issue"].unique()))
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.5)


# Training function with Early Stopping
def train_model_with_early_stopping(
    model, train_loader, test_loader, optimizer, device, num_epochs=10, patience=2, save_path="/home/ubuntu/NLP/roberta_model.pth"
):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=class_weights.to(device))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        val_preds, val_labels, val_logits = evaluate_model_with_metrics(model, test_loader, device, k=3)
        val_loss = torch.nn.functional.cross_entropy(
            torch.tensor(val_logits, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long)
        ).item()
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Model improved and saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


# Evaluation function
def evaluate_model_with_metrics(model, test_loader, device, k=3):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    if k > 1:
        top_k_acc = top_k_accuracy_score(all_labels, all_logits, k=k)
        print(f"Top-{k} Accuracy: {top_k_acc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    return all_preds, all_labels, all_logits


if os.path.exists("/home/ubuntu/NLP/roberta_model.pth"):
    model.load_state_dict(torch.load("/home/ubuntu/NLP/roberta_model.pth"))
    print("Model loaded successfully.")
else:
    train_model_with_early_stopping(model, train_loader, test_loader, optimizer, device)

all_preds, all_labels, all_logits = evaluate_model_with_metrics(model, test_loader, device, k=3)
