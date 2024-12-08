# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import pandas as pd
#
# # Dataset class
# class ComplaintDataset(Dataset):
#     def __init__(self, data, tokenizer, max_len):
#         self.input_ids = []
#         self.attention_masks = []
#         self.labels = data["Issue"]
#
#         for text in data["Processed Narrative"]:
#             encoded = tokenizer.encode_plus(
#                 text,
#                 max_length=max_len,
#                 truncation=True,
#                 padding="max_length",
#                 return_tensors="pt",
#             )
#             self.input_ids.append(encoded["input_ids"].squeeze(0))
#             self.attention_masks.append(encoded["attention_mask"].squeeze(0))
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         data = {
#             "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
#             "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
#             "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long),
#         }
#         return data
#
#
# # Load and preprocess data
# df = pd.read_csv("/home/ubuntu/final_df.csv")  # Adjust path if necessary
# df = df[df["Issue"].map(df["Issue"].value_counts()) > 5]  # Filtering rare classes
# df["Issue"] = df["Issue"].astype("category").cat.codes
#
# # Split data
# train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Issue"], random_state=42)
#
# # Tokenizer and Dataset
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# train_dataset = ComplaintDataset(train_data, tokenizer, max_len=128)
# test_dataset = ComplaintDataset(test_data, tokenizer, max_len=128)
#
# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16)
#
# # Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df["Issue"].unique()))
# model.to(device)
#
# # Optimizer
# optimizer = AdamW(model.parameters(), lr=2e-5)
#
# # Training function
# def train_model(model, train_loader, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     return total_loss / len(train_loader)
#
# # Evaluation function
# def evaluate_model(model, test_loader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#
#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             preds = torch.argmax(logits, axis=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#     return all_preds, all_labels
#
# # Training loop
# epochs = 3
# for epoch in range(epochs):
#     train_loss = train_model(model, train_loader, optimizer, device)
#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")
#
# # Evaluation
# all_preds, all_labels = evaluate_model(model, test_loader, device)
#
# # Classification report
# target_names = test_data["Issue"].astype("category").cat.categories
# print(classification_report(all_labels, all_preds, target_names=target_names))


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os

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
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long),
        }
        return data


# Load and preprocess data
df = pd.read_csv("/home/ubuntu/final_df.csv")  # Adjust path if necessary
df = df[df["Issue"].map(df["Issue"].value_counts()) > 5]  # Filtering rare classes
df["Issue"] = df["Issue"].astype("category").cat.codes

# Split data
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Issue"], random_state=42)

# Tokenizer and Dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = ComplaintDataset(train_data, tokenizer, max_len=128)
test_dataset = ComplaintDataset(test_data, tokenizer, max_len=128)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(df["Issue"].unique()))
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training function
def train_model(model, train_loader, optimizer, device, save_path="model.pth"):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)
    return total_loss / len(train_loader)

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
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
    return all_preds, all_labels

# Check if model exists
if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth"))
    print("Model loaded successfully.")
else:
    # Training loop
    epochs = 3
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluation
all_preds, all_labels = evaluate_model(model, test_loader, device)

# Ensure target names are strings
target_names = list(df["Issue"].astype("category").cat.categories.astype(str))

# Classification report
print(classification_report(all_labels, all_preds, target_names=target_names))
