#%%
import pandas as pd
final_df = pd.read_csv("/home/ubuntu/finalproject/Data/final_output.csv")

#%%
final_df.head()
#%%
response_counts = final_df['Company public response'].value_counts()



print("\nUnique values in 'public response' and their counts:")
print(response_counts)

# %%
mean_wordcount = final_df['word_count'].mean()
max_wordcount = final_df['word_count'].max()

print(f"Mean of wordcount: {mean_wordcount}")
print(f"Max of wordcount: {max_wordcount}")
# %%

count = len(final_df[final_df['word_count'] > 512])
print(count)
# %%
count = len(final_df[final_df['word_count'] < 512])
print(count)

#%%
final_df = final_df[final_df['word_count'] <= 512]
# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import pandas as pd

#%%\
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import pandas as pd
import time
from nltk.tokenize import sent_tokenize

# Load the DistilBART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (if available)
model.to(device)

def summarize_text(text):
    """
    Summarizes a single block of text using the DistilBART model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Move input tensor to the same device as the model (GPU or CPU)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_narrative(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints, and returns the updated DataFrame with the 'Summary' column.
    """
    summaries = []
    start_time = time.time()  # Track start time for progress updates
    total_records = len(final_df)
    checkpoint_interval = 5 * 60  # 5 minutes in seconds

    # Iterate over each row in the DataFrame
    for idx, narrative in enumerate(final_df['Processed Narrative']):
        # If the narrative has more than 1024 words, split and summarize
        if len(narrative.split()) > 512:
            # Split the narrative into sentences
            sentences = sent_tokenize(narrative)
            chunk = ""
            chunk_summaries = []

            # Accumulate sentences into chunks of up to 1024 words
            for sentence in sentences:
                if len(chunk.split()) + len(sentence.split()) <= 512:
                    chunk += " " + sentence
                else:
                    # Summarize the current chunk and reset the chunk for the next set of sentences
                    chunk_summaries.append(summarize_text(chunk))
                    chunk = sentence
            
            # Summarize the remaining chunk
            if chunk:
                chunk_summaries.append(summarize_text(chunk))

            # Join the chunk summaries into one final summary
            final_summary = " ".join(chunk_summaries)
            summaries.append(final_summary)
        else:
            # If narrative is <= 1024 words, summarize directly
            summaries.append(summarize_text(narrative))

        # Checkpoint: print every 5 minutes
        elapsed_time = time.time() - start_time
        if elapsed_time > checkpoint_interval:
            print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
            start_time = time.time()  # Reset start time after printing checkpoint

    # Add the summaries as a new column in the DataFrame
    final_df['Summary'] = summaries
    return final_df



# Apply the function to summarize narratives
final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])

# Save the model and tokenizer
output_dir = './distilbart_summary_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")


#%%
final_df.head()
final_df.to_csv("final_df.csv", index=False)

#%%
'''
#18HRS
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize

# Load the T5-base model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (if available)
model.to(device)

def summarize_text(text):
    """
    Summarizes a single block of text using the T5 model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move input tensor to the same device as the model (GPU or CPU)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_narrative(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints, and returns the updated DataFrame with the 'Summary' column.
    """
    summaries = []
    start_time = time.time()  # Track start time for progress updates
    total_records = len(final_df)
    checkpoint_interval = 5 * 60  # 5 minutes in seconds

    # Iterate over each row in the DataFrame
    for idx, narrative in enumerate(final_df['Processed Narrative']):
        # If the narrative has more than 512 words, split and summarize
        if len(narrative.split()) > 512:
            # Split the narrative into sentences
            sentences = sent_tokenize(narrative)
            chunk = ""
            chunk_summaries = []

            # Accumulate sentences into chunks of up to 512 words
            for sentence in sentences:
                if len(chunk.split()) + len(sentence.split()) <= 512:
                    chunk += " " + sentence
                else:
                    # Summarize the current chunk and reset the chunk for the next set of sentences
                    chunk_summaries.append(summarize_text(chunk))
                    chunk = sentence
            
            # Summarize the remaining chunk
            if chunk:
                chunk_summaries.append(summarize_text(chunk))

            # Join the chunk summaries into one final summary
            final_summary = " ".join(chunk_summaries)
            summaries.append(final_summary)
        else:
            # If narrative is <= 512 words, summarize directly
            summaries.append(summarize_text(narrative))

        # Checkpoint: print every 5 minutes
        elapsed_time = time.time() - start_time
        if elapsed_time > checkpoint_interval:
            print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
            start_time = time.time()  # Reset start time after printing checkpoint

    # Add the summaries as a new column in the DataFrame
    final_df['Summary'] = summaries
    return final_df

# Apply the function to summarize narratives
final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])

# Save the model and tokenizer
output_dir = './t5_base_summary_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")

'''

#%%
'''
import pandas as pd
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the DataFrame (assuming it's already in memory or loaded earlier)
# Example: final_df = pd.read_csv('path_to_your_csv_file.csv')


# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Function to summarize text using T5
def summarize_text(text, model, tokenizer, max_length=100, min_length=30):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Apply summarization with checkpoints
def summarize_with_checkpoints(df, model, tokenizer, checkpoint_interval=5):
    start_time = time.time()  # Start timer for progress tracking
    summaries = []  # Store the generated summaries
    total_records = len(df)
    
    for idx, row in df.iterrows():
        text = row['Processed Narrative']
        # Summarize only if the text is not null
        summary = summarize_text(text, model, tokenizer) if pd.notnull(text) else ""
        summaries.append(summary)
        
        # Add checkpoint logic
        elapsed_time = time.time() - start_time
        if elapsed_time > checkpoint_interval * 60:  # Check if 5 minutes have passed
            print(f"Checkpoint: {idx + 1}/{total_records} rows processed.")
            start_time = time.time()  # Reset the timer

    # Add the summaries as a new column
    df['Summary'] = summaries
    return df

# Process the DataFrame
checkpoint_interval = 5  # Interval in minutes for checkpoints
final_df = summarize_with_checkpoints(final_df, model, tokenizer, checkpoint_interval)

# Save the fine-tuned model
output_dir = './t5_finetuned_summary_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
'''

#%%
'''
#37HRS
import time
import torch
# Load the T5-large model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (if available)
model.to(device)

def summarize_text(text):
    """
    Summarizes a single block of text using the T5 model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move input tensor to the same device as the model (GPU or CPU)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_narrative(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints, and returns the updated DataFrame with the 'Summary' column.
    """
    summaries = []
    start_time = time.time()  # Track start time for progress updates
    total_records = len(final_df)
    checkpoint_interval = 5 * 60  # 5 minutes in seconds

    # Iterate over each row in the DataFrame
    for idx, narrative in enumerate(final_df['Processed Narrative']):
        # If the narrative has more than 512 words, split and summarize
        if len(narrative.split()) > 512:
            # Split the narrative into sentences
            sentences = sent_tokenize(narrative)
            chunk = ""
            chunk_summaries = []

            # Accumulate sentences into chunks of up to 512 words
            for sentence in sentences:
                if len(chunk.split()) + len(sentence.split()) <= 512:
                    chunk += " " + sentence
                else:
                    # Summarize the current chunk and reset the chunk for the next set of sentences
                    chunk_summaries.append(summarize_text(chunk))
                    chunk = sentence
            
            # Summarize the remaining chunk
            if chunk:
                chunk_summaries.append(summarize_text(chunk))

            # Join the chunk summaries into one final summary
            final_summary = " ".join(chunk_summaries)
            summaries.append(final_summary)
        else:
            # If narrative is <= 512 words, summarize directly
            summaries.append(summarize_text(narrative))

        # Checkpoint: print every 5 minutes
        elapsed_time = time.time() - start_time
        if elapsed_time > checkpoint_interval:
            print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
            start_time = time.time()  # Reset start time after printing checkpoint

    # Add the summaries as a new column in the DataFrame
    final_df['Summary'] = summaries
    return final_df



# Apply the function to summarize narratives
final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])

output_dir = './t5_finetuned_summary_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
'''

# %%
#T5Large
'''
import time
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set the start method for multiprocessing to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Load the T5-large model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (if available)
model.to(device)

def summarize_text(text):
    """
    Summarizes a single block of text using the T5 model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move input tensor to the same device as the model (GPU or CPU)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_narrative_for_chunk(narrative):
    """
    Helper function to process and summarize a single narrative chunk.
    """
    # If the narrative has more than 512 words, split and summarize
    if len(narrative.split()) > 512:
        # Split the narrative into sentences
        sentences = sent_tokenize(narrative)
        chunk = ""
        chunk_summaries = []

        # Accumulate sentences into chunks of up to 512 words
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= 512:
                chunk += " " + sentence
            else:
                # Summarize the current chunk and reset the chunk for the next set of sentences
                chunk_summaries.append(summarize_text(chunk))
                chunk = sentence
        
        # Summarize the remaining chunk
        if chunk:
            chunk_summaries.append(summarize_text(chunk))

        # Join the chunk summaries into one final summary
        final_summary = " ".join(chunk_summaries)
        return final_summary
    else:
        # If narrative is <= 512 words, summarize directly
        return summarize_text(narrative)

def process_narrative(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints, and returns the updated DataFrame with the 'Summary' column.
    """
    summaries = []
    start_time = time.time()  # Track start time for progress updates
    total_records = len(final_df)
    checkpoint_interval = 5 * 60  # 5 minutes in seconds

    # Create a ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = []
        for narrative in final_df['Processed Narrative']:
            # Submit each row for processing
            futures.append(executor.submit(process_narrative_for_chunk, narrative))
        
        # Collect results and track progress
        for idx, future in enumerate(as_completed(futures)):
            summaries.append(future.result())
            
            # Checkpoint: print every 5 minutes
            elapsed_time = time.time() - start_time
            if elapsed_time > checkpoint_interval:
                print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
                start_time = time.time()  # Reset start time after printing checkpoint

    # Add the summaries as a new column in the DataFrame
    final_df['Summary'] = summaries
    return final_df



# Apply the function to summarize narratives
final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])


# %%

import time
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Set the start method for multiprocessing to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Load the T5-base model and tokenizer (instead of T5-large)
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU (if available)
model.to(device)

def summarize_text(text):
    """
    Summarizes a single block of text using the T5 model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    
    # Move input tensor to the same device as the model (GPU or CPU)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_narrative_for_chunk(narrative):
    """
    Helper function to process and summarize a single narrative chunk.
    """
    if len(narrative.split()) > 512:
        sentences = sent_tokenize(narrative)
        chunk = ""
        chunk_summaries = []

        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= 512:
                chunk += " " + sentence
            else:
                chunk_summaries.append(summarize_text(chunk))
                chunk = sentence
        
        if chunk:
            chunk_summaries.append(summarize_text(chunk))

        final_summary = " ".join(chunk_summaries)
        return final_summary
    else:
        return summarize_text(narrative)

def process_narrative(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints, and returns the updated DataFrame with the 'Summary' column.
    """
    summaries = []
    start_time = time.time()
    total_records = len(final_df)
    checkpoint_interval = 5 * 60

    with ProcessPoolExecutor() as executor:
        futures = []
        for narrative in final_df['Processed Narrative']:
            futures.append(executor.submit(process_narrative_for_chunk, narrative))
        
        for idx, future in enumerate(as_completed(futures)):
            summaries.append(future.result())
            
            elapsed_time = time.time() - start_time
            if elapsed_time > checkpoint_interval:
                print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
                start_time = time.time()

    final_df['Summary'] = summaries
    return final_df

# Assuming `final_df` is already defined
final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])
'''
# %%
