"""
app.py - FastAPI application for SHL Assessment Recommendation Engine.

This module implements a REST API with FastAPI to serve the recommendation engine,
providing endpoints to recommend SHL assessments based on job descriptions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from recommender import SHLRecommender

# Initialize FastAPI app
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Path to data file - can be overridden with environment variable
DATA_FILE = os.getenv("SHL_DATA_FILE", "shl_assessments.json")

# Initialize recommender
recommender = None

class QueryInput(BaseModel):
    """Input model for recommendation request."""
    query: str = Field(..., description="Job description or natural language query")
    num_recommendations: int = Field(5, description="Number of recommendations to return (1-10)", ge=1, le=10)

class AssessmentOutput(BaseModel):
    """Output model for assessment recommendation."""
    id: str
    name: str
    url: str
    remote_support: bool
    adaptive_irt: bool
    test_types: List[str]
    duration: int
    similarity_score: float

class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint."""
    query: str
    recommendations: List[AssessmentOutput]

@app.on_event("startup")
async def startup_event():
    """Initialize the recommender on startup."""
    global recommender
    try:
        recommender = SHLRecommender(data_file=DATA_FILE)
    except Exception as e:
        print(f"Error initializing recommender: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SHL Assessment Recommendation API", 
            "endpoints": ["/recommend", "/health"]}

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(input_data: QueryInput):
    """
    Recommend SHL assessments based on a job description or query.
    
    Parameters:
    -----------
    input_data : QueryInput
        Input containing query and number of recommendations
        
    Returns:
    --------
    RecommendationResponse
        Response containing recommendations
    """
    global recommender
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    # Get recommendations
    recommendations = recommender.recommend(
        query=input_data.query,
        num_recommendations=input_data.num_recommendations
    )
    
    # Convert to response model
    assessment_outputs = []
    for rec in recommendations:
        assessment_outputs.append(AssessmentOutput(
            id=rec.get('id', ''),
            name=rec.get('name', ''),
            url=rec.get('url', ''),
            remote_support=rec.get('remote_support', False),
            adaptive_irt=rec.get('adaptive_irt', False),
            test_types=rec.get('test_types', []),
            duration=rec.get('duration', 0),
            similarity_score=rec.get('similarity_score', 0.0)
        ))
    
    return RecommendationResponse(
        query=input_data.query,
        recommendations=assessment_outputs
    )

@app.get("/recommend", response_model=RecommendationResponse)
async def recommend_get(
    query: str = Query(..., description="Job description or natural language query"),
    num_recommendations: int = Query(5, description="Number of recommendations to return (1-10)", ge=1, le=10)
):
    """
    Recommend SHL assessments based on a job description or query (GET method).
    
    Parameters:
    -----------
    query : str
        Job description or natural language query
    num_recommendations : int, optional
        Number of recommendations to return (default: 5)
        
    Returns:
    --------
    RecommendationResponse
        Response containing recommendations
    """
    input_data = QueryInput(query=query, num_recommendations=num_recommendations)
    return await recommend(input_data)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global recommender
    
    if recommender is None:
        return {"status": "error", "message": "Recommender not initialized"}
        
    if recommender.data_df is None or recommender.tfidf_matrix is None:
        return {"status": "error", "message": "Recommender data not loaded"}
    
    return {
        "status": "healthy", 
        "assessments_count": len(recommender.data_df) if recommender.data_df is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)