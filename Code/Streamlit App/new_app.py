import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BartTokenizer, BartForConditionalGeneration
import numpy as np

# Define Models for Sentiment, Summary, and Classification
class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sentiment(self, text):
        encoding = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[prediction.item()]

class TextSummarizer:
    def __init__(self, model_path):
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def summarize_text(self, text):
        encoding = self.tokenizer(text, truncation=True, padding=True, max_length=1024, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        with torch.no_grad():
            summary_ids = self.model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class ComplaintClassifier:
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_with_probability(self, inputs):
        encoding = self.tokenizer(inputs, truncation=True, padding=True, max_length=256, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        return prediction, probabilities[0].tolist()

# Sidebar for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1: Sentiment & Summary", "Page 2: Classification"])

# Load Models
@st.cache_resource
def load_models():
    sentiment_model = SentimentPredictor(
        model_path='/home/ubuntu/finalproject/models/sentiment_model',
        tokenizer_path='/home/ubuntu/finalproject/models/sentiment_tokenizer'
    )
    summary_model = TextSummarizer(
        model_path='/home/ubuntu/finalproject/code/bart_large_summary_model'
    )
    model_1 = ComplaintClassifier(
        model_path="/home/ubuntu/finalproject/code/bert_class_model_big",
        tokenizer_path="/home/ubuntu/finalproject/code/bert_class_model_big"
    )
    model_2 = ComplaintClassifier(
        model_path="/home/ubuntu/finalproject/code/bert_class_model_final",
        tokenizer_path="/home/ubuntu/finalproject/code/bert_class_model_final"
    )
    return sentiment_model, summary_model, model_1, model_2

sentiment_predictor, summarizer, model_1, model_2 = load_models()

if page == "Page 1: Sentiment & Summary":
    st.title("Customer Complaint Sentiment and Summary Analyzer")
    st.write("Enter your complaint text below to analyze its sentiment and generate a summary.")
    
    # Input fields
    narrative = st.text_area("Enter the Consumer Complaint Narrative:", height=150)
    
    if st.button("Analyze"):
        if narrative:
            sentiment = sentiment_predictor.predict_sentiment(narrative)
            summary = summarizer.summarize_text(narrative)
            
            st.subheader("Sentiment Analysis")
            st.write(f"**Sentiment**: {sentiment}")
            
            st.subheader("Summary")
            st.write(f"**Generated Summary**: {summary}")
            st.session_state.summary = summary

elif page == "Page 2: Classification":
    st.title("Company Response Classifier")
    st.write("Analyze consumer complaints and predict the company's response to the customer.")
    
    # Input fields
    narrative = st.text_area("Enter the Consumer Complaint Narrative:", height=150)
    company = st.text_input("Enter the Company Name:")
    product = st.text_input("Enter the Product:")
    issue = st.text_input("Enter the Issue:")
    
    if st.button("Classify"):
        if narrative:
            combined_input = " | ".join(filter(None, [
                f"Complaint Narrative: {narrative}",
                f"Company: {company}" if company else None,
                f"Product: {product}" if product else None,
                f"Issue: {issue}" if issue else None
            ]))
            
            summary = st.session_state.get("summary", "No summary generated in Page 1.")
            inputs = [combined_input, summary]
            
            labels = [
                "Closed with explanation",
                "Closed with non-monetary relief",
                "Closed with monetary relief",
                "Closed",
                "Untimely response"
            ]
            
            for i, text in enumerate(inputs, 1):
                pred_1, prob_1 = model_1.predict_with_probability(text)
                pred_2, prob_2 = model_2.predict_with_probability(text)
                
                st.subheader(f"Analysis {i}")
                st.write(f"**Input Text**: {text}")
                st.write(f"**Model 1 Prediction**: {labels[pred_1]} (Probabilities: {np.round(prob_1, 2)})")
                st.write(f"**Model 2 Prediction**: {labels[pred_2]} (Probabilities: {np.round(prob_2, 2)})")
