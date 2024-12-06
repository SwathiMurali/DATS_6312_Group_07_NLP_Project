#%%

import pandas as pd
#from deep_translator import GoogleTranslator
#%%
# Load your dataset
trans_data = pd.read_csv("/home/ubuntu/finalproject/Data/df_cleaned_preprocessed.csv")
#%%

# Display the first few rows to verify data loading
trans_data.head()
#%%
null_count = trans_data['Processed Narrative'].isna().sum()

print(f"Number of null values in 'Processed Narrative': {null_count}")

#%%
# Remove rows where 'Processed Narrative' column has NaN values
trans_data = trans_data.dropna(subset=['Processed Narrative'])

#%%

count_long_texts = trans_data['Processed Narrative'].apply(len).gt(4000).sum()
print(f"Number of rows with more than 4000 characters in 'Processed Narrative': {count_long_texts}")

#%%
count_long_texts2 = trans_data['Processed Narrative'].apply(len).gt(5000).sum()
print(f"Number of rows with more than 5000 characters in 'Processed Narrative': {count_long_texts2}")

#%%
count_long_texts3 = trans_data['Processed Narrative'].apply(len).gt(2000).sum()
print(f"Number of rows with more than 2000 characters in 'Processed Narrative': {count_long_texts3}")

#%%
trans_data['word_count'] = trans_data['Processed Narrative'].apply(lambda x: len(str(x).split()))

# Get the distribution of word counts
word_count_distribution = trans_data['word_count'].describe()

print(word_count_distribution)
#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load your dataset (ensure 'Processed Narrative' is the column containing text)
# Assuming trans_data is your DataFrame

# Calculate the word count for each row in 'Processed Narrative'
trans_data['word_count'] = trans_data['Processed Narrative'].apply(lambda x: len(str(x).split()))

# Define bins for word count ranges
bins = [0, 100, 300, 600, 1000, float('inf')]
labels = ['<100', '100-300', '301-600', '601-1000', '>1000']

# Create a new column 'word_count_range' to categorize each row
trans_data['word_count_range'] = pd.cut(trans_data['word_count'], bins=bins, labels=labels, right=False)

# Calculate the frequency of each word count range
word_count_freq = trans_data['word_count_range'].value_counts()

# Plot the frequency chart
plt.figure(figsize=(10, 6))
word_count_freq.plot(kind='bar', color='skyblue')
plt.title('Frequency of Text Lengths (Word Count Ranges)', fontsize=14)
plt.xlabel('Word Count Range', fontsize=12)
plt.ylabel('Frequency (in thousands)', fontsize=12)
plt.xticks(rotation=45)

# Format the y-axis labels to be in thousands
formatter = FuncFormatter(lambda x, _: f'{int(x/1000)}K' if x >= 1000 else f'{int(x)}')
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()


#%%


# Calculate total number of rows in the dataset
total_rows = len(trans_data)
print(f"Total rows in dataset: {total_rows}")
#%%

#Preprocessing 
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
    
    # Optional: Remove other unwanted characters (e.g., URLs, HTML tags, etc.)
    # text = re.sub(r'http\S+', '', text)  # Remove URLs if needed
    # text = re.sub(r'<.*?>', '', text)    # Remove HTML tags if needed
    
    return text


# Apply the cleaning function to the 'Consumer complaint narrative' column
trans_data['Processed Narrative'] = trans_data['Processed Narrative'].apply(clean_text)

#%%


#%%
trans_data.to_csv('df_preprocess_mj.csv', index=False)

#%%
import pandas as pd
trans_data=pd.read_csv('/home/ubuntu/finalproject/Data/df_preprocess_mj.csv')

#%%
missing_values = trans_data.isnull().sum()

# Display the missing values count
print(missing_values)
#%%
trans_data_cleaned = trans_data.dropna()

# Save the cleaned DataFrame to a CSV file
trans_data_cleaned.to_csv('df_preprocess_mj.csv', index=False)

#%%
'''
# Function to translate a batch of text to Spanish
def translate_batch_to_spanish(text_batch):
    try:
        # Translate each entry in the batch
        translations = [GoogleTranslator(source="auto", target="es").translate(text) for text in text_batch]
        return translations
    except Exception as e:
        print(f"Error translating batch: {e}")
        return [text for text in text_batch]  # Return original text in case of an error

#%%
# Function to apply translation in batches
def batch_process_translation(data, batch_size=100000):
    # List to store translated results
    translated_results = []
    
    # Process data in batches
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))  # End of the current batch
        batch = data[start:end]
        translated_batch = translate_batch_to_spanish(batch)  # Translate the batch
        translated_results.extend(translated_batch)  # Add translated batch to results

        # Print progress every 10,000 rows
        if (start // batch_size) % 10000 == 0:
            print(f"Processed {start + batch_size} rows out of {len(data)}")

    return translated_results

# Get the "Processed Narrative" column for translation
narrative_column = trans_data['Processed Narrative'].tolist()

# Translate in batches
translated_narratives = batch_process_translation(narrative_column)

# Add the translated column to the dataframe
trans_data['Espanol'] = translated_narratives

# Save the updated dataset
trans_data.to_csv("/home/ubuntu/finalproject/Data/df_translated_batch.csv", index=False)

# Display the first few rows of the updated dataset
print(trans_data.head())

# Print a message to confirm that the translation is complete
print("Translation completed and saved to 'df_translated_batch.csv'.")

#%%
# Track the number of successfully translated rows
translated_rows = trans_data['Espanol'].notna().sum()
print(f"Total translated rows: {translated_rows}")

#%%
'''

import pandas as pd
df_summary=pd.read_csv('/home/ubuntu/finalproject/Data/df_preprocess_mj.csv')
#%%
df_summary.head()

#%%
import pandas as pd

# Assuming df_summary is your original DataFrame


#%%
print
#%%

# Unique values and their counts for 'public response' column
response_counts = df_summary['Company public response'].value_counts()



print("\nUnique values in 'public response' and their counts:")
print(response_counts)


#%%
# Define the number of records to keep
n_records = 10000

# Filter 20k records for each specified value
subset1 = df_summary[df_summary['Company public response'] == "Company has responded to the consumer and the CFPB and chooses not to provide a public response"].sample(n=n_records, random_state=42)
subset2 = df_summary[df_summary['Company public response'] == "No Public Response"].sample(n=n_records, random_state=42)
subset3 = df_summary[df_summary['Company public response'] == "Company believes it acted appropriately as authorized by contract or law"].sample(n=n_records, random_state=42)

# Keep the rest of the records intact
remaining_records = df_summary[~df_summary['Company public response'].isin([
    "Company has responded to the consumer and the CFPB and chooses not to provide a public response",
    "No Public Response",
    "Company believes it acted appropriately as authorized by contract or law"
])]

# Concatenate all subsets to form the final DataFrame
final_df = pd.concat([subset1, subset2, subset3, remaining_records], ignore_index=True)

# Display the result
print(f"Total records in the final DataFrame: {len(final_df)}")

#%%
response_counts = final_df['Company public response'].value_counts()



print("\nUnique values in 'public response' and their counts:")
print(response_counts)

#%%
final_df.to_csv('final_output.csv', index=False)
