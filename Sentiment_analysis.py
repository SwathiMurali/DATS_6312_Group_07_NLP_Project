#%%

!pip install transformers
!pip install pandas



#%%
# Load the CSV file into a DataFrame
import pandas as pd
file_path = "/home/ubuntu/Project1/Data/df_cleaned_preprocessed.csv"
df = pd.read_csv(file_path)

#%%
import torch
print(torch.__version__)
#%%
!pip install torch torchvision torchaudio
#%%

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Check if CUDA is available and set device to GPU or CPU accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and tokenizer
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  # Move model to GPU

# Define a function for emotion analysis
def analyze_emotion(text):
    # Tokenize input and move tensors to the same device as the model (GPU or CPU)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    
    # Get model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the predicted label
    predicted_label = torch.argmax(probabilities, dim=1)
    emotion = model.config.id2label[predicted_label.item()]
    
    return emotion

# Apply emotion analysis to the 'Processed Narrative' column
df['Emotional Tone'] = df['Processed Narrative'].apply(analyze_emotion)


# Save the output or display the results
print(df[['Processed Narrative', 'Emotional Tone']])

#%%


