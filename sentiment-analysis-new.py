#%%
import pandas as pd

df = pd.read_csv('final_df.csv')
df.head()
# %%
df.info
# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(cleaned_tokens)

# Apply preprocessing to the 'Consumer complaint narrative' column
df['processed_text'] = df['Consumer complaint narrative'].apply(preprocess_text)

# Create a custom dataset class
class ComplaintDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.processed_text
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset
dataset = ComplaintDataset(df, tokenizer, max_len=128)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
# %%
print(df.shape)
print(df['processed_text'].head())
# %%
sample_item = dataset[0]
print(f"Sample item keys: {sample_item.keys()}")
print(f"Input IDs shape: {sample_item['ids'].shape}")
print(f"Attention Mask shape: {sample_item['mask'].shape}")
print(f"Token Type IDs shape: {sample_item['token_type_ids'].shape}")
# %%
for batch in data_loader:
    print(f"Batch size: {batch['ids'].shape[0]}")
    print(f"Input IDs shape: {batch['ids'].shape}")
    print(f"Attention Mask shape: {batch['mask'].shape}")
    print(f"Token Type IDs shape: {batch['token_type_ids'].shape}")
    break
# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probabilities, dim=-1)
    return sentiment.item()

# Apply sentiment prediction to the dataset
df['predicted_sentiment'] = df['processed_text'].apply(predict_sentiment)

# Map numerical labels to sentiment categories
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
df['predicted_sentiment'] = df['predicted_sentiment'].map(sentiment_map)
# %%
sentiment_counts = df['predicted_sentiment'].value_counts()
print(sentiment_counts)
print(sentiment_counts / len(df) * 100)
# %%
for sentiment in ['Negative', 'Neutral', 'Positive']:
    print(f"\n{sentiment} examples:")
    print(df[df['predicted_sentiment'] == sentiment]['processed_text'].head(3))
# %%
from sklearn.metrics import classification_report

if 'actual_sentiment' in df.columns:
    print(classification_report(df['actual_sentiment'], df['predicted_sentiment']))
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df['predicted_sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
# %%
sentiment_by_product = df.groupby('Product')['predicted_sentiment'].value_counts(normalize=True).unstack()
sentiment_by_product.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Sentiment Distribution by Product')
plt.xlabel('Product')
plt.ylabel('Proportion')
plt.legend(title='Sentiment')
plt.show()
# %%
