# Hotel Booking Analytics & QA System

An LLM-Powered system for analyzing hotel booking data and answering questions about bookings using RAG (Retrieval Augmented Generation).

## Features

- Analytics dashboard for hotel booking data
- RAG-powered question answering system
- FastAPI-based REST API
- SQLite database for data storage
- FAISS for efficient similarity search

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd hotel-booking-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
python main.py
```

The API will be available at:
- Local: http://127.0.0.1:3000
- Network: http://YOUR_IP:3000

## API Endpoints

1. `/docs` - Interactive API documentation (Swagger UI)
2. `/analytics` (POST) - Get analytics data
3. `/ask` (POST) - Ask questions about hotel bookings

## Project Structure

- `main.py` - FastAPI application and endpoints
- `preprocess.py` - Data preprocessing scripts
- `fine_tune.py` - Model fine-tuning scripts
- `database_setup.py` - Database initialization
- `data/` - Data directory
- `requirements.txt` - Project dependencies

## Technologies Used

- FastAPI
- SQLite
- FAISS
- Sentence Transformers
- Transformers (Hugging Face)
- Pandas
- NumPy 