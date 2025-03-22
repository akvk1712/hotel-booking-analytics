# Hotel Booking Analytics & QA System

LLM-Powered system for analyzing hotel booking data and providing intelligent answers to questions about bookings using RAG (Retrieval Augmented Generation) technology.

## 🚀 Features

- **Analytics Dashboard**: Comprehensive analysis of hotel booking patterns
- **RAG-Powered QA**: Intelligent question answering system using Retrieval Augmented Generation
- **FastAPI Backend**: Modern, high-performance REST API
- **SQLite Database**: Efficient data storage and retrieval
- **FAISS Integration**: Fast similarity search for relevant information retrieval
- **Interactive Documentation**: Swagger UI for easy API exploration

## 🛠️ Tech Stack

- **FastAPI**: Modern web framework for building APIs
- **SQLite**: Lightweight database for data storage
- **FAISS**: Efficient similarity search library
- **Sentence Transformers**: For generating text embeddings
- **Transformers**: Hugging Face's library for LLM integration
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/akvk1712/hotel-booking-analytics.git
cd hotel-booking-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the database setup:
```bash
python database_setup.py
```

## 🚀 Running the Application

1. Start the API server:
```bash
python main.py
```

2. Access the API:
- Local: http://127.0.0.1:3000
- Network: http://YOUR_IP:3000

## 📚 API Endpoints

### 1. Interactive Documentation
- **URL**: `/docs`
- **Method**: GET
- **Description**: Swagger UI for interactive API documentation and testing

### 2. Analytics Endpoint
- **URL**: `/analytics`
- **Method**: POST
- **Description**: Retrieve analytics data about hotel bookings
- **Response**: JSON object containing analytics metrics

### 3. Question Answering Endpoint
- **URL**: `/ask`
- **Method**: POST
- **Description**: Ask questions about hotel bookings
- **Request Body**:
```json
{
    "query": "What is the average length of stay?"
}
```
- **Response**: JSON object containing answer and relevant context

## 📁 Project Structure

```
hotel-booking-analytics/
├── main.py              # FastAPI application and endpoints
├── preprocess.py        # Data preprocessing scripts
├── fine_tune.py         # Model fine-tuning scripts
├── database_setup.py    # Database initialization
├── requirements.txt     # Project dependencies
├── README.md           # Project documentation
├── .gitignore          # Git ignore rules
└── data/               # Data directory
    └── processed_data.csv  # Processed hotel booking data
```

## 🔍 Example Usage

1. **Getting Analytics**:
```python
import requests

response = requests.post("http://localhost:3000/analytics")
analytics_data = response.json()
```

2. **Asking Questions**:
```python
import requests

query = {
    "query": "What are the most popular booking months?"
}
response = requests.post("http://localhost:3000/ask", json=query)
answer = response.json()
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

- **Akshath** - [akvk1712](https://github.com/akvk1712)

## 🙏 Acknowledgments

- FastAPI documentation
- Hugging Face Transformers library
- FAISS documentation
