import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

# Define the Classifier for Complaint Classification (Issue Classification)
class ComplaintClassifier:
    def __init__(self, model_path, tokenizer_path, num_labels):
        # Load the DistilBERT tokenizer and model architecture (no weights)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(tokenizer_path, num_labels=num_labels)  # Use correct number of labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Load the model's state_dict (weights) from the saved .pth file
        self.model.load_state_dict(torch.load(model_path))  # Load the saved weights

    def predict_with_probability(self, inputs):
        # Tokenize the inputs (text)
        encoding = self.tokenizer(inputs, truncation=True, padding=True, max_length=256, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Perform prediction with no gradient tracking (eval mode)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()

        return prediction, probabilities[0].tolist()

# Sidebar for Navigation
st.sidebar.title("Complaint Classification App")
page = st.sidebar.radio("Go to", ["Complaint Classification"])

# Load Model (cached to prevent reloading on every interaction)
@st.cache_resource
def load_models():
    # Specify the number of labels (105 for your case)
    model = ComplaintClassifier(
        model_path='/home/ubuntu/NLP/distilbert_model.pth',  # Path to your saved model weights
        tokenizer_path='distilbert-base-uncased',  # DistilBERT tokenizer
        num_labels=105  # Adjust to the correct number of classes in your model
    )
    return model

complaint_classifier = load_models()

if page == "Complaint Classification":
    st.title("Complaint Issue Classifier")
    st.write("Enter the complaint text below to classify the issue category.")
    
    # Input fields for complaint narrative
    narrative = st.text_area("Enter the Processed Complaint Narrative:", height=150)
    
    if st.button("Classify"):
        if narrative:
            # Use the classifier model to predict the issue category
            pred, prob = complaint_classifier.predict_with_probability(narrative)
            
            # Display the classification result
            st.subheader("Classification Result")
            st.write(f"Predicted Issue: {pred} (Probability: {np.round(prob[pred], 2)})")
