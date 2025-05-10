import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load dataset
df = pd.read_csv("phishing.csv", encoding='utf-8')
df.columns = df.columns.str.strip()
df.dropna(inplace=True) 

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

# GUI Code
def predict_email():
    email = email_input.get("1.0", "end-1c")  # Get the email text from the Text widget
    if email == '':
        messagebox.showwarning("Input Error", "Please enter email text to classify.")
        return
    
    email_clean = clean_text(email)
    email_vector = vectorizer.transform([email_clean])
    result = model.predict(email_vector)
    
    # Show the result in the result label
    prediction = "Phishing" if result[0] == 1 else "Legitimate"
    result_label.config(text=f"Prediction: {prediction}")

# Set up GUI window
root = tk.Tk()
root.title("Phishing Email Detection")

# Set window size and background color
root.geometry("600x500")
root.config(bg="#f4f4f9")

title_label = tk.Label(root, text="Phishing Email Detection", font=("Helvetica", 18, "bold"), bg="#f4f4f9", fg="#2c3e50")
title_label.pack(pady=20)

# Add Text widget for user input
email_input_label = tk.Label(root, text="Enter Email Text:", font=("Helvetica", 14), bg="#f4f4f9", fg="#34495e")
email_input_label.pack(pady=5)

email_input = tk.Text(root, height=8, width=45, font=("Helvetica", 12), wrap="word", bd=2, relief="solid")
email_input.pack(pady=10)

# Add a button to make prediction with styling
predict_button = tk.Button(root, text="Predict", font=("Helvetica", 14, "bold"), bg="#2980b9", fg="white", bd=0, relief="flat", width=20, height=2, command=predict_email)
predict_button.pack(pady=20)

result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14), bg="#f4f4f9", fg="#2c3e50")
result_label.pack(pady=10)

root.mainloop()
