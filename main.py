import faiss
import numpy as np
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import nest_asyncio
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load FAISS index & embeddings
index = faiss.read_index("faiss_index.bin")
embeddings = np.load("embeddings.npy")

# Load dataset for reference
df = pd.read_csv("data/processed_data.csv")

# Load Sentence Transformer for embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2")

# Use Pretrained GPT Model for Answer Generation
model_name = "distilgpt2"  # Using a smaller GPT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the pipeline
llm_pipeline = pipeline(
    "text-generation", 
    model=llm_model, 
    tokenizer=tokenizer, 
    device=0 if torch.cuda.is_available() else -1
)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Endpoint 1: Get Analytics from SQLite
@app.post("/analytics")
def get_analytics():
    conn = sqlite3.connect("hotel_analytics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analytics")
    results = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return results

# Endpoint 2: RAG-Powered Question Answering
@app.post("/ask")
def ask_question(request: QueryRequest = Body(...)):
    print("📥 Received Request:", request)
    print("🔍 Query Received:", request.query)
    
    query = request.query
    
    # Convert query to vector
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search FAISS index for relevant results
    _, indices = index.search(query_embedding, 5)  # Top 5 similar entries
    retrieved_texts = df.iloc[indices[0]]["query_text"].tolist()

    # Generate answer using Pretrained LLM
    context = " ".join(retrieved_texts)
    input_text = f"Context: {context} \nQuestion: {query} \nAnswer:"
    
    generated_answer = llm_pipeline(
        input_text, 
        max_length=100, 
        num_return_sequences=1
    )[0]["generated_text"]
    
    return {
        "retrieved_results": retrieved_texts, 
        "generated_answer": generated_answer
    }

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)