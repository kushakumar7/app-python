# Salesforce Chatbot API - Setup & Deployment Guide

This guide provides step-by-step instructions to configure, run, and deploy the Salesforce RAG (Retrieval-Augmented Generation) Chatbot.

---

## 📁 Project Structure

```text
project-root/
├── data/
│   └── Dataset.csv           # Dataset file
├── .env        # Environment variables (create this)
├── requirements.txt          # Project dependencies
├── main.py     # RAG logic & LLM configuration
└── app.py                    # FastAPI application
```

---

## ⚙️ Step 1: Environment Setup

Isolate your project dependencies by creating a virtual environment.

### Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

## 📦 Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Step 3: Configure Environment Variables

Create a `.env` file in the project root and add:

```env
# Required: Get this from your Groq Dashboard
API_KEY=your_actual_api_key_here

# Paths and Models
DATASET_PATH=data/Dataset.csv
LLM_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Tuning
RELEVANCE_THRESHOLD=1.0
```

---

## 🚀 Step 4: Run the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

---

## 🌐 API URL

```
http://127.0.0.1:8000
```

---

Your chatbot API should now be running locally. You can test endpoints using:
- Browser
- Postman
- Frontend integration

---