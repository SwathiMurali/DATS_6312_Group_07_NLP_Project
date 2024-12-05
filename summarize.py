#%%
import pandas as pd
df_summary=pd.read_csv('/home/ubuntu/finalproject/Data/df_preprocess_mj.csv')
#%%
train_data = df_summary[['Processed Narrative']]

# Create a target column that will have the same text (since we don't have summaries)
train_data['summary'] = train_data['Processed Narrative']

# Format the data for training
train_data = train_data[['Processed Narrative', 'summary']]
train_data = train_data.rename(columns={'Processed Narrative': 'input_text', 'summary': 'target_text'})

#%%
train_data.head()

#%%
print(train_data['input_text'].dtype)  # Check the data type of 'input_text' column
print(train_data['target_text'].dtype)  # Check the data type of 'target_text' column
print(train_data[['input_text', 'target_text']].isnull().sum())  # Check for missing values


#%%
'''
Tokenize the Data
You’ll tokenize both the input and the target text. In the absence of actual summaries, the model will learn to generate a condensed or meaningful version of the input text.
'''
!pip install sentencepiece
from transformers import T5Tokenizer
import sentencepiece
!pip install datasets
from datasets import Dataset
from transformers import T5Tokenizer


#%%
!pip install torch torchvision torchaudio sentencepiece datasets
import torch
from datasets import Dataset
from transformers import T5Tokenizer


#%%
import torch
print(torch.__version__)
print(torch.cuda.is_available())

#%%
from datasets import Dataset
from datasets import load_dataset
# Initialize the tokenizer
from transformers import T5Tokenizer

# Initialize the tokenizer
from transformers import T5Tokenizer
from datasets import Dataset

# Initialize the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-large')

# Tokenize the input and target text
def tokenize_data(example):
    # Tokenizing both input and target
    input_encodings = tokenizer(example['input_text'], truncation=True, padding='max_length', max_length=512, return_attention_mask=True)
    target_encodings = tokenizer(example['target_text'], truncation=True, padding='max_length', max_length=128, return_attention_mask=True)
    
    return {**input_encodings, **target_encodings}

# Apply the tokenization to the dataset using map()
train_data = train_data.map(tokenize_data, batched=True)




# %%
# Display the first 5 rows of the dataset
train_data.select(range(5))

# %%
!pip install datasets
from datasets import Dataset

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
# %%
