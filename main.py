import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================
# 1. LOAD CONFIGURATION
# =========================
load_dotenv()

API_KEY = os.getenv("API_KEY")
DATASET_PATH = os.getenv("DATASET_PATH", "Dataset - Sheet1 (1).csv")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
# Default to 1.0 if not specified in .env
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 1.0)) 

if not API_KEY:
    raise ValueError("API_KEY not found in .env")

client = Groq(api_key=API_KEY)

# =========================
# 2. DATA PREPARATION
# =========================
# Load and clean the dataset as per original requirements
df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

def clean_text(text):
    return str(text).replace("\n", " ").strip()

df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)
df["combined"] = df["question"] + " " + df["answer"]

# =========================
# 3. EMBEDDING & FAISS INDEX
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(df["combined"].tolist()).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# =========================
# 4. UTILITY FUNCTIONS
# =========================

def generate_response(prompt, max_tokens=200):
    """Core LLM call using Groq."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def retrieve(query, k=1):
    """Searches the FAISS index for the most relevant context."""
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "question": df.iloc[idx]["question"],
            "answer": df.iloc[idx]["answer"],
            "score": float(distances[0][rank])
        })
    return results

def is_relevant(retrieved_data, threshold=RELEVANCE_THRESHOLD):
    """Checks if the top match is within the acceptable distance."""
    if not retrieved_data:
        return False
    best_score = retrieved_data[0]["score"]
    print(f"Best Match Score: {best_score}") 
    return best_score < threshold

# =========================
# 5. HANDLERS & CLASSIFICATION
# =========================

def handle_domain_query(query, retrieved_data):
    """Answers Salesforce-specific questions using retrieved context."""
    context = "\n".join([
        f"Q: {item['question']}\nA: {item['answer']}"
        for item in retrieved_data
    ])
    
    prompt = f"""
    You are a Salesforce assistant.
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say: "I don't know based on available data."
    If answer contains multiple steps, use bullet points.

    Context:
    {context}

    Question: {query}
    Answer:
    """
    return generate_response(prompt)

def handle_general_query(query):
    """Answers general knowledge questions."""
    prompt = f"""
    You are a helpful assistant. Answer the question clearly and simply.
    Question: {query}
    Answer:
    """
    return generate_response(prompt)

def classify_query(query):
    """Determines if the query is DOMAIN (Salesforce) or GENERAL."""
    prompt = f"""
    You are an intelligent query classifier. 
    Classify the following query as DOMAIN (Salesforce/Internal) or GENERAL.
    Query: {query}
    Output: Return ONLY the word DOMAIN or GENERAL.
    """
    result = generate_response(prompt, max_tokens=10).upper()
    return "GENERAL" if "GENERAL" in result else "DOMAIN"

# =========================
# 6. CHATBOT LOGIC
# =========================

def chatbot(query):
    """Main routing logic for the chatbot."""
    query_type = classify_query(query)

    if query_type == "GENERAL":
        answer = handle_general_query(query)
        return {
            "route": "GENERAL",
            "score": None,
            "answer": answer,
            "retrieved": []
        }

    retrieved = retrieve(query, k=1)

    if is_relevant(retrieved):
        answer = handle_domain_query(query, retrieved)
        route = "DATASET"
    else:
        answer = "I don't know based on available data."
        route = "NO_MATCH"

    return {
        "route": route,
        "score": retrieved[0]["score"] if retrieved else None,
        "answer": answer,
        "retrieved": retrieved
    }