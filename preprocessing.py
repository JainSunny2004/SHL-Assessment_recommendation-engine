"""
preprocessing.py - Data loading and preprocessing for SHL recommendation engine.

This module handles loading the JSON data from SHL's product catalog and
performs necessary preprocessing steps to prepare it for the recommendation system.
"""

import json
import pandas as pd
from typing import Dict, List, Union
import re

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load SHL assessment data from JSON file into a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON file containing SHL assessment data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the preprocessed SHL assessment data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure all required columns are present
        required_columns = ['id', 'name', 'url', 'remote_support', 'adaptive_irt', 
                           'test_types', 'description', 'duration']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in data: {missing_columns}")
            
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the SHL assessment data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the raw SHL assessment data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the preprocessed SHL assessment data
    """
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original DataFrame
    processed_df = df.copy()
    
    # Ensure required columns are present
    if 'description' not in processed_df.columns:
        processed_df['description'] = ''
        
    # Clean text data
    processed_df['description'] = processed_df['description'].apply(clean_text)
    
    # Handle test_types - convert to string format if it's a list
    if 'test_types' in processed_df.columns:
        processed_df['test_types_str'] = processed_df['test_types'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
    
    # Convert boolean columns to Yes/No for display
    if 'remote_support' in processed_df.columns:
        processed_df['remote_support_display'] = processed_df['remote_support'].apply(
            lambda x: 'Yes' if x else 'No'
        )
    
    if 'adaptive_irt' in processed_df.columns:
        processed_df['adaptive_irt_display'] = processed_df['adaptive_irt'].apply(
            lambda x: 'Yes' if x else 'No'
        )
    
    return processed_df

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data for NLP tasks.
    
    Parameters:
    -----------
    text : str
        Text to clean
        
    Returns:
    --------
    str
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Keep hyphens and certain special characters that might be important
    # but remove other punctuation
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_features_for_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and prepare features used for recommendation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the preprocessed SHL assessment data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing features for recommendation
    """
    if df.empty:
        return df
        
    features_df = df.copy()
    
    # Combine description with test types for better content matching
    if 'test_types_str' in features_df.columns and 'description' in features_df.columns:
        features_df['combined_features'] = features_df['description'] + ' ' + features_df['test_types_str']
    elif 'description' in features_df.columns:
        features_df['combined_features'] = features_df['description']
    
    return features_df

if __name__ == "__main__":
    # Example usage
    try:
        # Modify this path to point to your JSON file
        file_path = "shl_assessments.json"
        df = load_data(file_path)
        if not df.empty:
            processed_df = preprocess_data(df)
            features_df = get_features_for_recommendation(processed_df)
            print(f"Loaded {len(features_df)} assessments.")
            print(f"Columns: {features_df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {str(e)}")