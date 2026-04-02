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

# ✅ Robust dataset path (works everywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(BASE_DIR, "Dataset.csv"))

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 1.0))

# Validate API key
if not API_KEY:
    raise ValueError("API_KEY not found in .env")

client = Groq(api_key=API_KEY)

# =========================
# 2. LOAD DATA
# =========================
print(f"Loading dataset from: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
df = df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)

def clean_text(text):
    return str(text).replace("\n", " ").strip()

df["question"] = df["question"].apply(clean_text)
df["answer"] = df["answer"].apply(clean_text)
df["combined"] = df["question"] + " " + df["answer"]

print(f"Dataset loaded successfully. Total records: {len(df)}")

# =========================
# 3. EMBEDDINGS + FAISS
# =========================
print("Loading embedding model...")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = embed_model.encode(df["combined"].tolist()).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built successfully")

# =========================
# 4. CORE FUNCTIONS
# =========================

def generate_response(prompt, max_tokens=200):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def retrieve(query, k=1):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "question": df.iloc[idx]["question"],
            "answer": df.iloc[idx]["answer"],
            "score": float(distances[0][i])
        })
    return results


def is_relevant(results):
    if not results:
        return False
    score = results[0]["score"]
    print(f"Match Score: {score}")
    return score < RELEVANCE_THRESHOLD


# =========================
# 5. CHATBOT LOGIC
# =========================

def chatbot(query):
    retrieved = retrieve(query)

    if is_relevant(retrieved):
        context = "\n".join([
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in retrieved
        ])

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.
If answer not found, say: "I don't know based on available data."

Context:
{context}

Question: {query}
Answer:
"""
        answer = generate_response(prompt)
        route = "DATASET"
    else:
        answer = "I don't know based on available data."
        route = "NO_MATCH"

    return {
        "route": route,
        "answer": answer,
        "score": retrieved[0]["score"] if retrieved else None
    }
