# phishing_detector.py

import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("phishing_emails.csv")  # Rename if needed

# Basic text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['text'] = df['text'].astype(str).apply(clean_text)

# Split data
X = df['text']
y = df['label']  # Make sure column is named 'label' (0: legit, 1: phishing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Predict & evaluate
y_pred = model.predict(X_test_vectors)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# Try a sample
while True:
    email = input("\nEnter email text to test (or type 'exit'):\n")
    if email == 'exit':
        break
    email_clean = clean_text(email)
    email_vector = vectorizer.transform([email_clean])
    result = model.predict(email_vector)
    print("Prediction:", "Phishing" if result[0] == 1 else "Legitimate")
