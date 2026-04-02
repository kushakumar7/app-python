import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from main import chatbot

app = FastAPI(title="Salesforce Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    result = chatbot(request.query)
    return result

@app.get("/")
def home():
    return {"status": "API is running", "environment": os.getenv("ENVIRONMENT", "production")}