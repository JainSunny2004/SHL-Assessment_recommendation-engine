# SHL Assessment Recommendation Engine

A content-based recommendation system that matches job descriptions with relevant SHL assessments using NLP techniques.

## Features

- **Content-Based Matching**: Uses TF-IDF vectorization and cosine similarity to find the most relevant assessments
- **FastAPI Backend**: High-performance API with automatic documentation
- **Streamlit UI**: Interactive user interface for testing and demonstration
- **Evaluation Framework**: Built-in evaluation using standard IR metrics (Recall@k, MAP@k)
- **Docker Support**: Easy deployment with Docker

## Live Demo & API

- **Demo URL**: [https://shl-recommender-demo.example.com](https://shl-recommender-demo.example.com)
- **API Endpoint**: [https://shl-recommender-api.example.com/recommend](https://shl-recommender-api.example.com/recommend)
- **API Documentation**: [https://shl-recommender-api.example.com/docs](https://shl-recommender-api.example.com/docs)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shl-recommender.git
   cd shl-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare your data (replace with your actual assessment data):
   ```bash
   # Sample data is included in the repo as shl_assessments.json
   ```

### Running the Application

#### Using Python directly:

1. Start the API server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Start the Streamlit UI (in a separate terminal):
   ```bash
   streamlit run streamlit_app.py
   ```


```

## Evaluation

The system can be evaluated using standard information retrieval metrics:

```bash
python run_evaluation.py --test test_queries1.json --detailed
```


## API Usage

### Using cURL:

```bash
curl -X GET "http://localhost:8000/recommend?query=Looking%20for%20Java%20developers&num_recommendations=5"
```

### Using Python requests:

```python
import requests

response = requests.get(
    "http://localhost:8000/recommend",
    params={
        "query": "Looking for Java developers",
        "num_recommendations": 5
    }
)
print(response.json())
```

## Architecture

```
┌───────────────┐     ┌───────────────┐      ┌───────────────┐
│ Streamlit UI  │────▶│ FastAPI       │────▶│ Recommender   │
│ (User         │◀────│ (API          │◀────│ (Core         │
│  Interface)   │     │  Endpoints)   │      │  Engine)      │
└───────────────┘     └───────────────┘      └───────┬───────┘
                                                   │
                                                   ▼
                                             ┌───────────────┐
                                             │ TF-IDF +      │
                                             │ Cosine        │
                                             │ Similarity    │
                                             └───────────────┘
```

## Project Structure

```
shl-recommender/
├── app.py                # FastAPI application
├── recommender.py        # Core recommendation engine
├── preprocessing.py      # Data preprocessing utilities
├── evaluate.py           # Evaluation framework
├── run_evaluation.py     # Evaluation script
├── streamlit_app.py      # Streamlit UI application
├── shl_assessments.json  # Sample assessment data
├── test_queries1.json    # Test queries with expected results
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # Project documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SHL for providing the assessment catalog data
- The FastAPI and Streamlit teams for their excellent frameworks