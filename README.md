# 🛡️ PhishGuard — AI Phishing Email Detector

A full-stack cybersecurity project that uses Machine Learning to detect phishing emails in real time.

Built with **FastAPI**, **React.js**, **scikit-learn**, and **Tailwind CSS**.

---

## 🔍 Features

- ✅ Real-time phishing detection using an ensemble ML model
- ✅ Confidence score with visual meter
- ✅ Red flags breakdown (urgency words, IP URLs, domain mismatch, etc.)
- ✅ Feature-by-feature analysis panel
- ✅ Scan history for the session
- ✅ Sample phishing and legit email loader
- ✅ OWASP-aligned checks (XSS indicators, obfuscation, suspicious URLs)

---

## 🧠 ML Model

- **Algorithm:** Ensemble — Random Forest + Gradient Boosting + Logistic Regression (soft voting)
- **Text features:** TF-IDF (unigrams + bigrams, 3000 features)
- **Numeric features:** 15 engineered signals (urgency words, caps, exclamations, URL type, domain mismatch, etc.)
- **Dataset:** 2000 synthetic labeled samples (50/50 phishing/legit)
- **Accuracy:** ~99%+ on test set

---

## 🗂️ Project Structure

phishing-detector/
├── backend/
│ ├── model/
│ │ ├── train_model.py # ML training script
│ │ └── phishing_model.pkl # Saved model (auto-generated)
│ ├── utils/
│ │ └── feature_extractor.py # Email feature engineering
│ └── main.py # FastAPI server
├── frontend/
│ ├── public/
│ │ └── index.html
│ └── src/
│ ├── components/
│ │ └── Navbar.jsx
│ ├── pages/
│ │ ├── AnalyzerPage.jsx
│ │ ├── ResultPage.jsx
│ │ └── HistoryPage.jsx
│ ├── utils/
│ │ └── api.js
│ ├── App.jsx
│ ├── index.js
│ └── index.css
├── dataset/
│ ├── generate_dataset.py
│ └── phishing_dataset.csv # Auto-generated
├── .gitignore
└── README.md

---

## 🚀 Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
```

### 2. Backend setup

```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib scipy

# Generate dataset
cd dataset
python generate_dataset.py

# Train the model
cd ../backend/model
python train_model.py

# Start the API server
cd ..
uvicorn main:app --reload
```

### 3. Frontend setup

```bash
cd frontend
npm install
npm start
```

### 4. Open the app

http://localhost:3000

---

## 🛠️ Tech Stack

| Layer    | Technology                            |
| -------- | ------------------------------------- |
| Frontend | React.js, Tailwind CSS                |
| Backend  | Python, FastAPI, Uvicorn              |
| ML Model | scikit-learn, TF-IDF, Ensemble Voting |
| Data     | Pandas, NumPy, SciPy                  |

---

## 👨‍💻 Author

**Bhavesh Kerkar**  
BSc IT — PTVA's Sathaye College, Mumbai  
[GitHub](https://github.com/bhaveshkerkar) · [LinkedIn](https://linkedin.com/in/bhaveshkerkar)
