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

#%%
#BART MOdel-LLM
from transformers import pipeline
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=r"max_length is set to \d+, but your input_length is only \d+.")

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)  # Use device=0 for GPU

def summarize_narratives(final_df):
    """
    Processes the 'Processed Narrative' column of the DataFrame,
    summarizes long complaints using the BART model.
    """
    summaries = []
    start_time = time.time()  # Track start time for progress updates
    total_records = len(final_df)
    checkpoint_interval = 5 * 60  # 5 minutes in seconds

    # Iterate over each row in the DataFrame
    for idx, narrative in enumerate(final_df['Processed Narrative']):
        # Determine the max length based on narrative length
        narrative_length = len(narrative.split())
        if narrative_length < 10:
            summaries.append("")
            continue
        if narrative_length < 50:
            min_length = narrative_length
        else:
            min_length = 50

        if narrative_length < 120:
            max_length = narrative_length
        else:
            max_length = 120
        
        # Summarize with adjusted max_length
        summary = summarizer(narrative, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']
        summaries.append(summary)

        # Checkpoint: print every 5 minutes
        elapsed_time = time.time() - start_time
        if elapsed_time > checkpoint_interval:
            print(f"Checkpoint: {idx + 1}/{total_records} records summarized.")
            start_time = time.time()  # Reset start time after printing checkpoint

    # Add the summaries as a new column in the DataFrame
    final_df['Summary'] = summaries
    return final_df

# Process the narratives
final_df = summarize_narratives(final_df)



# Save the updated DataFrame to a CSV file
output_csv_path = 'BART.csv'
final_df.to_csv(output_csv_path, index=False)

print(f"Summarized data saved to {output_csv_path}")

# Save the model and tokenizer
output_dir = './bart_large_summary_model'
summarizer.model.save_pretrained(output_dir)
summarizer.tokenizer.save_pretrained(output_dir)

#%%
#Evaluation

from rouge_score import rouge_scorer
import pandas as pd

# Function to calculate ROUGE scores
def calculate_rouge(df, column_generated, column_reference, num_rows=1000):
    """
    Calculates ROUGE scores for the generated and reference summaries.
    
    Args:
    - df: DataFrame containing the data.
    - column_generated: Column name for the generated summaries.
    - column_reference: Column name for the reference texts.
    - num_rows: Number of rows to evaluate (default is 1000).

    Returns:
    - A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores (precision, recall, and F1).
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    # Iterate through the first `num_rows` rows
    for idx, row in df.head(num_rows).iterrows():
        reference = row[column_reference]  # Original text
        generated = row[column_generated]  # Summarized text

        # Skip if any text is missing
        if not reference or not generated:
            continue

        # Calculate ROUGE scores
        score = scorer.score(reference, generated)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    # Average scores
    avg_scores = {key: sum(values) / len(values) if values else 0 for key, values in scores.items()}
    return avg_scores

# Load the data
# Ensure 'Processed Narrative' is the reference column and 'Summary' is the generated column
eval_bart_df = pd.read_csv('/home/ubuntu/finalproject/Data/BART.csv')

# Calculate ROUGE for the first 1000 rows
rouge_scores = calculate_rouge(final_df, column_generated='Summary', column_reference='Processed Narrative', num_rows=1000)

# Print ROUGE scores
print("ROUGE Scores for the first 1000 rows:")
for key, value in rouge_scores.items():
    print(f"{key}: {value:.4f}")

#%%
#ROUGE Scores for the first 1000 rows:
'''
rouge1: 0.7361
rouge2: 0.7020
rougeL: 0.6813

'''
# %%


#%%
'''very poor results'''
'''
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
import pandas as pd

final_df=pd.read_csv("/home/ubuntu/finalproject/Data/final_df.csv")

#%%
import matplotlib.pyplot as plt

# Calculate word count for each summary
final_df['WordCount'] = final_df['Processed Narrative'].apply(lambda x: len(x.split()))

# Display basic statistics about the word counts
print("Word Count Statistics:")
print(final_df['WordCount'].describe())

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.hist(final_df['WordCount'], bins=30, color='skyblue', edgecolor='black')
plt.title('Word Count Distribution in Summaries', fontsize=16)
plt.xlabel('Word Count', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


#%%

from bert_score import score

def evaluate_bertscore(generated, reference):
    """
    Computes BERTScore between generated and reference summaries.
    """
    P, R, F1 = score([generated], [reference], lang="en", verbose=True)
    return {"Precision": P.item(), "Recall": R.item(), "F1": F1.item()}

reference_summary = final_df['Processed Narrative']
generated_summary = final_df['Summary']
# Example
bertscore_result = evaluate_bertscore(generated_summary, reference_summary)
print(bertscore_result)

#%%
print(f"Length of reference_summary: {len(reference_summary)}")
print(f"Length of generated_summary: {len(generated_summary)}")

#%%
final_df = final_df.dropna(subset=['Processed Narrative', 'Summary'])
reference_summary = final_df['Processed Narrative'].tolist()
generated_summary = final_df['Summary'].tolist()
#%%
final_df.head()




#%%
#To remove repitions
def remove_repetitions(text):
    """
    Removes repeated sentences or phrases from the given text.
    """
    sentences = sent_tokenize(text)  # Tokenize into sentences
    unique_sentences = list(dict.fromkeys(sentences))  # Remove duplicates while maintaining order
    return " ".join(unique_sentences)  # Reconstruct the text

def preprocess_narrative(final_df):
    """
    Preprocesses the 'Processed Narrative' column by removing repetitions.
    Returns the DataFrame with the cleaned column.
    """
    final_df['Processed Narrative'] = final_df['Processed Narrative'].apply(remove_repetitions)
    return final_df

# Preprocess the DataFrame to remove repetitions
final_df = preprocess_narrative(final_df)

#%%

#18HRS
'''
'''
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

final_df = process_narrative(final_df)

# Display the DataFrame with the new "Summary" column
print(final_df[['Processed Narrative', 'Summary']])

# Save the updated DataFrame to a CSV file
output_csv_path = 't5translate.csv'
final_df.to_csv(output_csv_path, index=False)

print(f"Summarized data saved to {output_csv_path}")

# Save the model and tokenizer
output_dir = './t5_base_summary_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
'''


#%%

#%%
'''
37 hrs to run T5 large'''