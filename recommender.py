"""
recommender.py - Core recommendation logic for SHL assessment recommendation engine.

This module implements content-based recommendation using TF-IDF on assessment descriptions
and other relevant features to find the most relevant SHL assessments for job descriptions.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import re
from preprocessing import load_data, preprocess_data, get_features_for_recommendation

class SHLRecommender:
    """
    A content-based recommendation system for SHL assessments.
    Uses TF-IDF vectorization and cosine similarity to match job descriptions
    with relevant SHL assessments.
    """
    
    def __init__(self, data_file: str = None, data_df: pd.DataFrame = None):
        """
        Initialize the SHL recommender.
        
        Parameters:
        -----------
        data_file : str, optional
            Path to the JSON file containing SHL assessment data
        data_df : pd.DataFrame, optional
            DataFrame containing SHL assessment data (used if data_file is None)
        """
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.data_df = None
        
        if data_df is not None:
            self.load_dataframe(data_df)
        elif data_file is not None:
            self.load_data(data_file)
    
    def load_data(self, data_file: str) -> None:
        """
        Load and preprocess data from a JSON file.
        
        Parameters:
        -----------
        data_file : str
            Path to the JSON file containing SHL assessment data
        """
        raw_df = load_data(data_file)
        if not raw_df.empty:
            processed_df = preprocess_data(raw_df)
            features_df = get_features_for_recommendation(processed_df)
            self.load_dataframe(features_df)
    
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load and preprocess data from a DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing SHL assessment data
        """
        self.data_df = df
        
        # Initialize and fit TF-IDF vectorizer
        if 'combined_features' in self.data_df.columns or 'description' in self.data_df.columns:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 3)  # Include unigrams, bigrams, and trigrams for better matching
            )
            
            # Create TF-IDF matrix
            # Use combined_features if available, otherwise use description
            feature_column = 'combined_features' if 'combined_features' in self.data_df.columns else 'description'
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.data_df[feature_column].fillna('')
            )
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query or job description.
        
        Parameters:
        -----------
        query : str
            User query or job description
            
        Returns:
        --------
        str
            Preprocessed query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Keep hyphens and certain special characters that might be important
        # but remove other punctuation
        query = re.sub(r'[^\w\s\-]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def recommend(self, query: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend SHL assessments based on a job description or query.
        
        Parameters:
        -----------
        query : str
            User query or job description
        num_recommendations : int, optional
            Number of recommendations to return (default: 5)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of recommended assessments with their details
        """
        if self.data_df is None or self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Transform query using the TF-IDF vectorizer
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity between query and assessments
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top N similar assessment indices
        top_indices = cosine_similarities.argsort()[-num_recommendations:][::-1]
        
        # Create list of recommended assessments
        recommendations = []
        for idx in top_indices:
            if cosine_similarities[idx] > 0:  # Only include relevant assessments
                assessment = self.data_df.iloc[idx]
                
                recommendation = {
                    'id': assessment.get('id', ''),
                    'name': assessment.get('name', ''),
                    'url': assessment.get('url', ''),
                    'remote_support': assessment.get('remote_support', False),
                    'adaptive_irt': assessment.get('adaptive_irt', False),
                    'test_types': assessment.get('test_types', []),
                    'description': assessment.get('description', ''),
                    'duration': assessment.get('duration', 0),
                    'similarity_score': float(cosine_similarities[idx])
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def evaluate(self, test_queries: List[Dict[str, Any]], 
                 k: int = 3) -> Tuple[float, float]:
        """
        Evaluate the recommender using Mean Recall@k and MAP@k.
        
        Parameters:
        -----------
        test_queries : List[Dict[str, Any]]
            List of test queries, each with a query and expected assessment names
        k : int, optional
            Number of recommendations to consider (default: 3)
            
        Returns:
        --------
        Tuple[float, float]
            Mean Recall@k and MAP@k scores
        """
        if not test_queries:
            return 0.0, 0.0
            
        recalls = []
        avg_precisions = []
        
        for test_item in test_queries:
            query = test_item.get('query', '')
            
            # Check if using the new format with expected_assessments or old format with relevant_ids
            if 'expected_assessments' in test_item:
                relevant_items = test_item.get('expected_assessments', [])
                field_to_match = 'name'
            else:
                relevant_items = test_item.get('relevant_ids', [])
                field_to_match = 'id'
            
            if not query or not relevant_items:
                continue
                
            # Get recommendations
            recommendations = self.recommend(query, num_recommendations=k)
            recommended_items = [rec.get(field_to_match, '') for rec in recommendations]
            
            # Calculate Recall@k
            relevant_recommended = set(recommended_items) & set(relevant_items)
            recall = len(relevant_recommended) / len(relevant_items) if relevant_items else 0
            recalls.append(recall)
            
            # Calculate Average Precision for MAP
            avg_precision = 0
            relevant_count = 0
            
            for i, rec_item in enumerate(recommended_items):
                if rec_item in relevant_items:
                    relevant_count += 1
                    precision_at_i = relevant_count / (i + 1)
                    avg_precision += precision_at_i
            
            avg_precision = avg_precision / len(relevant_items) if relevant_items else 0
            avg_precisions.append(avg_precision)
        
        # Calculate Mean Recall@k and MAP@k
        mean_recall_at_k = sum(recalls) / len(recalls) if recalls else 0
        map_at_k = sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0
        
        return mean_recall_at_k, map_at_k

if __name__ == "__main__":
    # Example usage
    try:
        recommender = SHLRecommender("shl_assessments.json")
        sample_query = "Looking for a sales manager who can lead a team and develop client relationships"
        recommendations = recommender.recommend(sample_query, num_recommendations=3)
        
        print(f"Recommendations for query: '{sample_query}'")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} (Similarity: {rec['similarity_score']:.4f})")
            print(f"   Test Types: {', '.join(rec['test_types'])}")
            print(f"   Duration: {rec['duration']} minutes")
            print(f"   Remote Support: {'Yes' if rec['remote_support'] else 'No'}")
            print()
            
    except Exception as e:
        print(f"Error: {str(e)}")