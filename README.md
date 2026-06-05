# 🩺 Smart Healthcare Chatbot

A bilingual AI-powered healthcare consultation assistant that combines Machine Learning, Rule-Based Systems, and Google Gemini LLM to provide structured health guidance in both English and Hindi.

## 🚀 Live Demo

**Hugging Face Deployment:**
https://anjaliup89-smarthealthcarebot.hf.space

## 📌 Features

* 🌐 Bilingual Healthcare Guidance (English & Hindi)
* 🤖 Google Gemini LLM Integration
* 🧠 Machine Learning-based Clinical Intent Detection
* 🚨 Emergency Symptom Recognition
* 📊 Confidence Score & Telemetry Metrics
* 🔊 Text-to-Speech Voice Output
* 🏥 Structured Healthcare Reports
* ⚡ Real-time Symptom Analysis

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-Learn
* Logistic Regression
* TF-IDF Vectorizer
* Google Gemini API
* Deep Translator
* gTTS
* Docker
* Hugging Face Spaces

## 🏗️ Project Architecture

User Input
↓
Language Translation
↓
Rule-Based Emergency Detection
↓
ML Intent Classification (Logistic Regression + TF-IDF)
↓
Google Gemini LLM Processing
↓
English & Hindi Healthcare Report Generation
↓
Audio Output (gTTS)

## 📂 Project Structure

```text
SmartHealthcarebot/
│
├── streamlit_app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anjaliup63/Smart-Healthcare-Chatbot-using-llm.git
cd Smart-Healthcare-Chatbot-using-llm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run streamlit_app.py
```

## 📊 Machine Learning Model

* TF-IDF Vectorization
* Logistic Regression Classifier
* Clinical Intent Prediction
* Emergency Keyword Detection

## ⚠️ Disclaimer

This application is designed for educational and informational purposes only. It does not provide medical diagnosis, treatment recommendations, or professional healthcare advice. Always consult a qualified healthcare professional for medical concerns.

## 👩‍💻 Author

**Anjali Upadhyay**

B.Tech Computer Science Engineer
AI/ML Enthusiast | Generative AI | Machine Learning | NLP

GitHub: https://github.com/anjaliup63

