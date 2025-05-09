import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("phishing.csv", encoding='utf-8')
df.columns = df.columns.str.strip()
df.dropna(inplace=True)# Remove extra spaces just in case

# Basic text preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['EmailText'] = df['EmailText'].astype(str).apply(clean_text)
X = df['EmailText']
y = df['Label']  # Make sure column is named 'label' (0: legit, 1: phishing)

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
