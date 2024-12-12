
#%%
'''Loaded dataset from an other file which was generated from data_inspection_EDA.ipynb'''
trans_data = pd.read_csv("/home/ubuntu/finalproject/Data/df_cleaned_preprocessed.csv")
#%%

trans_data.head()
#%%

'''Removing null values'''
null_count = trans_data['Processed Narrative'].isna().sum()

print(f"Number of null values in 'Processed Narrative': {null_count}")

#%%
# Remove rows where 'Processed Narrative' column has NaN values
trans_data = trans_data.dropna(subset=['Processed Narrative'])


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

'''Some EDA on Counting words in each narrative'''
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


import pandas as pd
df_summary=pd.read_csv('/home/ubuntu/finalproject/Data/df_preprocess_mj.csv')
#%%
df_summary.head()


"Let's clean the data and reduce the size, Since it taking long time to train my translation model"
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

'''
Summarized the columns to check if my classification is better working with the Consumer Narrative or Summary of Consumer Narrative
'''