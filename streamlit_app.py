import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

class SentimentPredictor:
    def __init__(self, model_path='sentiment_model/sentiment_model', tokenizer_path='sentiment_tokenizer/sentiment_tokenizer'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict_sentiment(self, text):
        # Tokenize and prepare input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)
        
        # Map prediction to sentiment
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[prediction.item()]

# Streamlit UI
st.title("Customer Complaint Sentiment Analyzer")
st.write("Enter your complaint text below to analyze its sentiment.")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize sentiment predictor
@st.cache_resource
def load_model():
    return SentimentPredictor()

predictor = load_model()

# Text input
user_input = st.text_area("Enter your complaint:", height=100)

if st.button("Analyze Sentiment"):
    if user_input:
        # Get prediction
        sentiment = predictor.predict_sentiment(user_input)
        
        # Add to chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", f"Sentiment: {sentiment}"))
        
        # Clear input
        st.rerun()

# Display chat history
st.write("### Chat History")
for role, message in st.session_state.chat_history:
    if role == "user":
        st.write(f"👤 **You**: {message}")
    else:
        st.write(f"🤖 **Bot**: {message}")
        if "Sentiment: " in message:
            sentiment = message.split(": ")[1]
            if sentiment == "Positive":
                st.success("😊 Positive Sentiment Detected")
            elif sentiment == "Negative":
                st.error("😔 Negative Sentiment Detected")
            else:
                st.info("😐 Neutral Sentiment Detected")
    st.write("---")

# Add some helpful information
st.sidebar.title("About")
st.sidebar.write("""
This application analyzes the sentiment of customer complaints using a fine-tuned BERT model.
The model classifies text into three categories:
- 😊 Positive
- 😐 Neutral
- 😔 Negative
""")