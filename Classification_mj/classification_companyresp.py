#%%
import pandas as pd
df_class=pd.read_csv("/home/ubuntu/finalproject/Data/BART.csv")
#%%
# This will also return the number of rows in the DataFrame
num_rows = len(df_class)
print(f"Number of rows: {num_rows}")
#%%
unique_counts = df_class["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)
#%%
# Remove rows where "Company response to consumer" is "In progress"
df_class_cleaned = df_class[df_class["Company response to consumer"] != "In progress"]


# Display the cleaned DataFrame
print(f"Number of rows after cleaning: {len(df_class_cleaned)}")
#%%
unique_counts = df_class_cleaned["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)

#%%
df_full=pd.read_csv("/home/ubuntu/finalproject/Data/df_cleaned_preprocessed.csv")
#%%

num_rows = len(df_full)
print(f"Number of rows: {num_rows}")

# Remove rows with any null (NaN) values
df_full_cleaned = df_full.dropna()

# Alternatively, if you want to remove columns with any null values:
# df_full_cleaned = df_full.dropna(axis=1)

#%%
import pandas as pd
df_class=pd.read_csv("/home/ubuntu/finalproject/Data/BART.csv")
#%%
# This will also return the number of rows in the DataFrame
num_rows = len(df_class)
print(f"Number of rows: {num_rows}")
#%%
unique_counts = df_class["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)
#%%
# Remove rows where "Company response to consumer" is "In progress"
df_class_cleaned = df_class[df_class["Company response to consumer"] != "In progress"]


# Display the cleaned DataFrame
print(f"Number of rows after cleaning: {len(df_class_cleaned)}")
#%%
unique_counts = df_class_cleaned["Company response to consumer"].value_counts()

# Display the result
print(unique_counts)

#%%
print(np.unique(df_class_cleaned['Company response to consumer']))

#%%

#%%
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
    df['text'] = df['Summary'] + " " + df['Product'] + " " + df['Issue'] + " " + df['Company']
    
    df['text'] = df['text'].fillna("").astype(str)
    # Encode 'target' (the 'Company response to consumer' column) into numeric labels
    label_encoder = LabelEncoder()
    df['target'] = label_encoder.fit_transform(df['Company response to consumer'])
    
    return df[['text', 'target']], label_encoder  # Return the label encoder for later decoding


def tokenize_data(tokenizer, texts, labels, max_length=256):
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
    Trains and evaluates the BERT model with progress updates.
    """
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Training
        model.train()
        train_loss = 0
        correct_train_predictions = 0
        total_train_predictions = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader_train, desc="Training Batches")):
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
            
            # Show batch loss every 10 batches
            #if (batch_idx + 1) % 10 == 0:
                #batch_accuracy = correct_train_predictions / total_train_predictions
               # print(f"Batch {batch_idx + 1}: Loss={loss.item():.4f}, Accuracy={batch_accuracy:.4f}")
        
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
    df = df_class_cleaned

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
    train_and_evaluate(model, dataloader_train, dataloader_val, optimizer, scheduler, epochs=epochs)

    # Decode predictions (optional)
    decoded_labels = decode_predictions(model, dataloader_val, label_encoder)
    output_dir = './bert_class_model'
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Sample Predictions:", decoded_labels[:5])


if __name__ == "__main__":
    main()

# %%
