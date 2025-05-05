"""
streamlit_app.py - Streamlit UI for SHL Assessment Recommendation Engine.

This module provides a simple web interface using Streamlit to allow users
to input job descriptions and view recommended SHL assessments.
"""

import streamlit as st
import pandas as pd
import json
import os
import requests
from typing import List, Dict, Any, Optional

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")  # Default to localhost
DEFAULT_EXAMPLE = """
Job Title: Sales Account Manager

Responsibilities:
- Develop and maintain client relationships to drive business growth 
- Create and implement account strategies for key clients
- Monitor sales performance metrics and provide regular reports
- Negotiate contracts and close deals with clients
- Collaborate with internal teams to ensure client satisfaction

Requirements:
- 3+ years of experience in B2B sales
- Strong communication and relationship-building skills
- Goal-oriented with the ability to work under pressure
- Analytical thinking and problem-solving abilities
"""

# Page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to call the API
def get_recommendations(query: str, num_recommendations: int = 5) -> Dict[str, Any]:
    """
    Call the recommendation API to get assessment recommendations.
    
    Parameters:
    -----------
    query : str
        Job description or natural language query
    num_recommendations : int, optional
        Number of recommendations to return (default: 5)
        
    Returns:
    --------
    Dict[str, Any]
        API response with recommendations
    """
    try:
        # Check if we should use local recommender or API
        if os.path.exists("shl_assessments.json") and not API_URL.startswith("http"):
            # Import here to avoid import errors if not needed
            from recommender import SHLRecommender
            
            # Use local recommender
            recommender = SHLRecommender(data_file="shl_assessments.json")
            recommendations = recommender.recommend(query, num_recommendations=num_recommendations)
            
            return {
                "query": query,
                "recommendations": recommendations
            }
        else:
            # Call API
            response = requests.get(
                f"{API_URL}/recommend",
                params={"query": query, "num_recommendations": num_recommendations},
                timeout=30
            )
            
            if response.status_code != 200:
                st.error(f"Error from API: {response.status_code} - {response.text}")
                return {"query": query, "recommendations": []}
                
            return response.json()
            
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return {"query": query, "recommendations": []}

# Helper function to render recommendations
def render_recommendations(response: Dict[str, Any]) -> None:
    """
    Render recommendation results in Streamlit.
    
    Parameters:
    -----------
    response : Dict[str, Any]
        API response with recommendations
    """
    if not response or "recommendations" not in response:
        st.warning("No recommendations available.")
        return
        
    recommendations = response["recommendations"]
    
    if not recommendations:
        st.warning("No matching assessments found for this job description.")
        return
    
    st.subheader("Recommended SHL Assessments")
    
    # Create a table for the recommendations
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {i}. [{rec['name']}]({rec['url']})")
                
                # Display similarity score as a progress bar
                similarity = rec.get('similarity_score', 0)
                st.progress(min(similarity, 1.0))
                st.caption(f"Relevance Score: {similarity:.2f}")
                
                # Check if description exists and display it
                if 'description' in rec and rec['description']:
                    with st.expander("Assessment Description"):
                        st.write(rec['description'])
            
            with col2:
                # Display assessment details
                st.markdown("**Details:**")
                st.markdown(f"⏱️ Duration: {rec.get('duration', 'N/A')} minutes")
                st.markdown(f"🌐 Remote Testing: {'✅' if rec.get('remote_support', False) else '❌'}")
                st.markdown(f"🔄 Adaptive/IRT: {'✅' if rec.get('adaptive_irt', False) else '❌'}")
                
                # Display test types
                if 'test_types' in rec and rec['test_types']:
                    st.markdown("**Test Types:**")
                    test_types = rec['test_types']
                    if isinstance(test_types, list):
                        for test_type in test_types:
                            st.markdown(f"- {test_type}")
                    else:
                        st.markdown(f"- {test_types}")
        
        st.divider()

# Main application layout
def main():
    """Main Streamlit application."""
    st.title("SHL Assessment Recommendation Engine")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool helps you find the most relevant SHL assessments based on your job description or query.
        
        Simply enter a job description or specific requirements, and the system will recommend 
        the most suitable SHL assessments from their product catalog.
        """)
        
        st.header("Settings")
        num_recommendations = st.slider(
            "Number of recommendations", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Select how many assessment recommendations you want to see"
        )
        
        st.header("Examples")
        if st.button("Load Sales Manager Example"):
            st.session_state.job_description = DEFAULT_EXAMPLE
    
    # Main content
    st.header("Enter Job Description")
    
    # Initialize session state for job description if not exists
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    
    # Job description input
    job_description = st.text_area(
        "Enter a job description or specific requirements",
        value=st.session_state.job_description,
        height=200,
        placeholder="Paste job description here..."
    )
    
    # Update session state when input changes
    if job_description != st.session_state.job_description:
        st.session_state.job_description = job_description
    
    # Submit button
    if st.button("Get Recommendations") or ('recommendations' in st.session_state and job_description == st.session_state.last_query):
        if not job_description.strip():
            st.warning("Please enter a job description or query.")
        else:
            with st.spinner("Analyzing job description and finding relevant assessments..."):
                # Get recommendations
                response = get_recommendations(job_description, num_recommendations)
                st.session_state.recommendations = response
                st.session_state.last_query = job_description
                
                # Render recommendations
                render_recommendations(response)
    
    # Show footer
    st.markdown("---")
    st.caption("SHL Assessment Recommendation Engine | Powered by TF-IDF + Cosine Similarity")

if __name__ == "__main__":
    main()