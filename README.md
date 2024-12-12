# DATS 6312: Group 07 NLP Project:
## CUSTOMER COMPLAINT ANAYSIS AND PREDICTION SYSTEM

### Team Members:
Bhumika Mallikarjun Horapet  
Manoj Padala  
Swathi Murali  
Swathi K R    

## Overview of the Project:
The project focuses on leveraging advanced Natural Language Processing (NLP) and Machine Learning techniques to process and derive meaningful insights from this dataset. The four primary tasks that were undertaken are:  
•	Complaint Classification: Classifying consumer complaints into relevant categories based on the type of issue reported, enabling companies to effectively address different customer concerns.  
•	Summarization: Automating the process of summarizing lengthy consumer narratives into concise, informative summaries. This allows businesses to quickly grasp the key points of customer feedback without sifting through extensive text.  
•	Sentiment Analysis: Evaluating the sentiment expressed in consumer narratives to understand the emotional tone behind customer complaints or feedback. This helps in determining customer satisfaction levels and identifying areas for improvement.  
•	Company Response Classification: Assessing the quality of company responses to consumer complaints, focusing on aspects like timeliness and relevance. This task is crucial for evaluating the effectiveness of customer service practices and improving response strategies.  

## Dataset Source: 
Consumer Complaint Database - https://www.consumerfinance.gov/data-research/consumer-complaints/

## Models Used:
1. DistilBERT
2. RoBERTa
3. T5-Small
4. BART-Large-CNN

## Framework and Libraries:
1. PyTorch
2. Hugging Face Transformers
3. Scikit-learn
4. Pandas

## Code Structure:

1. Data Preprocessing  
Input: Raw consumer complaint data with various features (e.g., complaint narrative, company response).
Steps:
Clean text by removing special characters, punctuation, and numbers.
Tokenize text using model-specific tokenizers (e.g., DistilBERT, RoBERTa).
Handle missing data and imbalanced classes using sampling techniques.
Core Tasks and Models

2. Complaint Classification:
Classifies consumer complaints into predefined categories using models like DistilBERT and RoBERTa.

3. Summarization:
Summarizes lengthy complaint narratives using BART and T5 models.

4. Sentiment Analysis:
Identifies sentiment (positive, neutral, negative) using fine-tuned BERT-based models.
Company Response Classification:
Assesses response timeliness and relevance using models like DistilBERT.

5. Model Training and Tuning: 
Models are fine-tuned on labeled datasets with a focus on accuracy, F1-score, and ROUGE scores.
Techniques such as cross-validation, weighted loss functions, and learning rate scheduling ensure robust performance.

6. Evaluation and Deployment: 
Each model's output is evaluated against test data using relevant metrics (e.g., F1-score, ROUGE).
Deployed models predict complaint categories, summarize narratives, and analyze sentiments in real-time


## Results:



