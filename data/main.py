import faiss
import numpy as np
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import nest_asyncio
import uvicorn
import socket

# Initialize FastAPI
app = FastAPI(title="Hotel Booking Analytics API")

try:
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
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

# Define request model
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return {
        "status": "API is running",
        "endpoints": ["/docs", "/analytics", "/ask"],
        "local_ip": local_ip,
        "access_url": f"http://{local_ip}:3000"
    }

# Endpoint 1: Get Analytics from SQLite
@app.post("/analytics")
def get_analytics():
    try:
        conn = sqlite3.connect("hotel_analytics.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analytics")
        results = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: RAG-Powered Question Answering
@app.post("/ask")
def ask_question(request: QueryRequest = Body(...)):
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    nest_asyncio.apply()
    print("Starting server...")
    # Get the local IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\nAccess the API at: http://{local_ip}:3000")
    print(f"Or locally at: http://127.0.0.1:3000\n")
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True) 