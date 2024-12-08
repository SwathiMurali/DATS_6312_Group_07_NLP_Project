# Import libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the preprocessed dataset
df = pd.read_csv('final_df.csv')  # Ensure this is the output of Preprocess.py
df = df[['Processed Narrative', 'Issue']].dropna()

# Step 2: Inspect dataset
print(f"Dataset Shape: {df.shape}")
print(f"Sample Data:\n{df.head()}")

# Step 3: Text Vectorization with TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,  # Adjust based on your dataset size
    ngram_range=(1, 2),
    stop_words='english'
)
X = vectorizer.fit_transform(df['Processed Narrative'])
y = df['Issue']

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model and vectorizer
import pickle
with open('complaint_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved.")

