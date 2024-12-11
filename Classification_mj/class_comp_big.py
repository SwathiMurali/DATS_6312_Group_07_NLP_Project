

#%%
import pandas as pd
df_full=pd.read_csv("/Data/df_cleaned_preprocessed.csv")
#%%

num_rows = len(df_full)
print(f"Number of rows: {num_rows}")
#%%
# Remove rows with any null (NaN) values
df_full_cleaned = df_full.dropna()

# Alternatively, if you want to remove columns with any null values:
df_full_cleaned = df_full.dropna(axis=1)



#%%
# This will also return the number of rows in the DataFrame
num_rows = len(df_full_cleaned)
print(f"Number of rows: {num_rows}")
#%%
unique_counts = df_full_cleaned["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)
#%%
# Remove rows where "Company response to consumer" is "In progress"
df_full_cleaned = df_full_cleaned[df_full_cleaned["Company response to consumer"] != "In progress"]


# Display the cleaned DataFrame
print(f"Number of rows after cleaning: {len(df_full_cleaned)}")
#%%
unique_counts = df_full_cleaned["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)

#%%
df_full_cleaned = df_full_cleaned[df_full_cleaned['Complaint Length'] <= 512]
#%%
import pandas as pd

# Example: Assuming df_full_cleaned is your DataFrame
# Count the occurrences of each category
counts = df_full_cleaned["Company response to consumer"].value_counts()

# Get the categories with counts > 100,000
categories_to_reduce = counts[counts > 100000].index

# Filter the data for categories exceeding 100,000
filtered_df = df_full_cleaned[
    ~df_full_cleaned["Company response to consumer"].isin(categories_to_reduce)
]

# Display unique counts in the reduced data
reduced_counts = filtered_df["Company response to consumer"].value_counts()
print(reduced_counts)

#%%
# Initialize an empty list to store reduced data for each category
reduced_data = []

for category in df_full_cleaned["Company response to consumer"].unique():
    category_data = df_full_cleaned[
        df_full_cleaned["Company response to consumer"] == category
    ]
    if len(category_data) > 100000:
        # Randomly sample 100,000 rows
        category_data = category_data.sample(n=100000, random_state=42)
    reduced_data.append(category_data)

# Concatenate all the reduced data
filtered_df = pd.concat(reduced_data)

# Display the new unique counts
reduced_counts = filtered_df["Company response to consumer"].value_counts()
#%%
print(filtered_df.head())
#%%
unique_counts = filtered_df["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)



#%%
import re

def clean_text(text):
    # Remove leading/trailing whitespaces
    text = text.strip()
    
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove legal references (e.g., "123 USC")
    text = re.sub(r'\b\d{1,5}\s+usc\b', '', text, flags=re.IGNORECASE)
    
    # Remove 'xxxx' patterns
    text = re.sub(r'\b[x]{2,}\b', '', text)
    
    return text


# Apply the cleaning function to the 'Consumer complaint narrative' column
filtered_df['Consumer complaint narrative'] = filtered_df['Consumer complaint narrative'].apply(clean_text)

#%%import pandas as pd



#model for 3M records
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# Determine the device (GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


def prepare_data(df):
    """
    Prepares the DataFrame for training by creating a text column and encoding labels.
    """
    # Ensure required columns are present
    required_columns = ['Consumer complaint narrative', 'Product', 'Issue', 'Company', 'Company response to consumer']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing one or more required columns: {required_columns}")

    df['text'] = (
        df['Consumer complaint narrative'].fillna("") + " " +
        df['Product'].fillna("") + " " +
        df['Issue'].fillna("") + " " +
        df['Company'].fillna("")
    )
    label_encoder = LabelEncoder()
    if df['Company response to consumer'].isnull().any():
        raise ValueError("Missing values detected in target column.")
    df['target'] = label_encoder.fit_transform(df['Company response to consumer'])
    return df[['text', 'target']], label_encoder


def tokenize_data(tokenizer, texts, labels, max_length=512):
    """
    Tokenizes input texts and creates TensorDataset.
    """
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        padding="max_length",
        truncation=True,  # Truncate sequences longer than max_length
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_masks, labels)


def train_and_evaluate(model, dataloader_train, dataloader_val, optimizer, tokenizer, scheduler, loss_fn, epochs=4):
    """
    Trains and evaluates the BERT model and computes F1-score and accuracy.
    """
    best_val_f1 = 0.0
    best_model_dir = './best_model'

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        # Training loop
        model.train()
        train_loss = 0
        train_predictions = []
        train_labels = []

        for batch in tqdm(dataloader_train, desc="Training Batches"):
            batch_input_ids, batch_attention_masks, batch_labels = [item.to(device) for item in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits

            train_loss += loss.item()
            train_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(batch_labels.cpu().numpy())

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_loss / len(dataloader_train)
        train_f1 = f1_score(train_labels, train_predictions, average="weighted")
        train_accuracy = accuracy_score(train_labels, train_predictions)
        print(f"Training Loss: {avg_train_loss:.4f}, F1-Score: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader_val, desc="Validation Batches"):
                batch_input_ids, batch_attention_masks, batch_labels = [item.to(device) for item in batch]
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks)
                logits = outputs.logits
                loss = loss_fn(logits, batch_labels)

                val_loss += loss.item()
                val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(batch_labels.cpu().numpy())

        avg_val_loss = val_loss / len(dataloader_val)
        val_f1 = f1_score(val_labels, val_predictions, average="weighted")
        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Loss: {avg_val_loss:.4f}, F1-Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")
        print("-" * 30)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"New best model saved with F1-Score: {best_val_f1:.4f}")


def main():
    # Load dataset
    df = filtered_df
    df_prepared, label_encoder = prepare_data(df)

    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df_prepared['target']),
        y=df_prepared['target']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        df_prepared['text'].values,
        df_prepared['target'].values,
        test_size=0.2,
        random_state=42,
        stratify=df_prepared['target'].values
    )

    # Load tokenizer and tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    dataset_train = tokenize_data(tokenizer, X_train, y_train)
    dataset_val = tokenize_data(tokenizer, X_val, y_val)

    # Load model
    num_labels = len(label_encoder.classes_)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    # DataLoader
    batch_size = 16
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs)

    # Train and evaluate
    train_and_evaluate(model, dataloader_train, dataloader_val, optimizer, tokenizer, scheduler, loss_fn, epochs=epochs)

    # Save final model
    output_dir = './bert_class_model_big'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()



#%%
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
def prepare_data(df):
    """
    Prepares the DataFrame for training by creating a text column and encoding labels.
    """
    # Combine relevant columns into a single text column
    df['text'] = df['Consumer complaint narrative'] + " " + df['Product'] + " " + df['Issue'] + " " + df['Company']
    
    df['text'] = df['text'].fillna("").astype(str)
    # Encode 'target' (the 'Company response to consumer' column) into numeric labels
    label_encoder = LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['Company response to consumer'])
    
    return df[['text', 'target']], label_encoder  # Return the label encoder for later decoding


def tokenize_data(tokenizer, texts, labels, max_length=512):
    """
    Tokenizes input texts and creates TensorDataset.
    """
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_masks, labels)

def train_and_evaluate(model, dataloader_train, dataloader_val, optimizer, scheduler, epochs=4):
    """
    Trains and evaluates the BERT model with progress updates and saves the best-performing model.
    """
    loss_fn = CrossEntropyLoss()
    best_val_accuracy = 0.0  # Track the best validation accuracy
    best_model_dir = './best_model'  # Directory to save the best model

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0
        
        for batch in tqdm(dataloader_train, desc="Training Batches"):
            batch_input_ids, batch_attention_masks, batch_labels = [item.to("cuda") for item in batch]

            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Update training loss
            train_loss += loss.item()
            
            # Calculate training accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_train_predictions += (predictions == batch_labels).sum().item()
            total_train_predictions += batch_labels.size(0)
            
            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # End of epoch training metrics
        avg_train_loss = train_loss / len(dataloader_train)
        train_accuracy = correct_train_predictions / total_train_predictions
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct_val_predictions = 0
        total_val_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader_val, desc="Validation Batches"):
                batch_input_ids, batch_attention_masks, batch_labels = [item.to("cuda") for item in batch]
                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
                loss = outputs.loss
                logits = outputs.logits

                # Update validation loss
                val_loss += loss.item()
                
                # Calculate validation accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_val_predictions += (predictions == batch_labels).sum().item()
                total_val_predictions += batch_labels.size(0)
        
        # End of epoch validation metrics
        avg_val_loss = val_loss / len(dataloader_val)
        val_accuracy = correct_val_predictions / total_val_predictions
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print("-" * 30)
        
        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            print(f"New best model saved with accuracy: {best_val_accuracy:.4f}")



def decode_predictions(model, dataloader, label_encoder):
    """
    Decodes predictions from the model into human-readable labels.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_masks = [item.to("cuda") for item in batch[:2]]
            logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks).logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    decoded_labels = label_encoder.inverse_transform(predictions)
    return decoded_labels


def main():
    # Load your dataset
    df = filtered_df

    # Prepare data
    df_prepared, label_encoder = prepare_data(df)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        df_prepared['text'].values,
        df_prepared['target'].values,
        test_size=0.15,
        random_state=42,
        stratify=df_prepared['target'].values
    )

    # Load tokenizer and tokenize data
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    #tokenizer = BertTokenizer.from_pretrained('/home/ubuntu/finalproject/models/sentiment_tokenizer', do_lower_case=True)
    dataset_train = tokenize_data(tokenizer, X_train, y_train)
    dataset_val = tokenize_data(tokenizer, X_val, y_val)

    # Load model
    num_labels = len(label_encoder.classes_)
    '''
    model = BertForSequenceClassification.from_pretrained(
    '/home/ubuntu/finalproject/models/sentiment_model',
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
    ).to("cuda")
    '''
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    ).to("cuda")  # Move model to GPU
    


    # DataLoader
    batch_size = 16
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train) * epochs)

    # Train and evaluate
    train_and_evaluate(model, dataloader_train, dataloader_val, optimizer, scheduler, , tokenizer, epochs=epochs)

    # Decode predictions (optional)
    decoded_labels = decode_predictions(model, dataloader_val, label_encoder)
    output_dir = './bert_class_model_big'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Sample Predictions:", decoded_labels[:5])


if __name__ == "__main__":
    main()


'''