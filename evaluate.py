"""
evaluate.py - Evaluation module for SHL recommendation engine.

This module provides functionality to evaluate the recommendation system
using Mean Recall@k and Mean Average Precision (MAP)@k metrics.
"""

import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Tuple
from recommender import SHLRecommender

def load_test_queries(test_file: str) -> List[Dict[str, Any]]:
    """
    Load test queries from a JSON file.
    
    Parameters:
    -----------
    test_file : str
        Path to the JSON file containing test queries
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of test queries with query text and expected assessment names
    """
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            test_queries = json.load(f)
        
        # Validate test queries format
        for i, query in enumerate(test_queries):
            if 'query' not in query or 'expected_assessments' not in query:
                print(f"Warning: Test query at index {i} is missing 'query' or 'expected_assessments' field")
        
        return test_queries
    
    except FileNotFoundError:
        print(f"Error: Test queries file {test_file} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {test_file}.")
        return []
    except Exception as e:
        print(f"Error loading test queries: {str(e)}")
        return []

def calculate_metrics(recommender: SHLRecommender, test_queries: List[Dict[str, Any]], 
                     k: int = 3) -> Tuple[float, float]:
    """
    Calculate evaluation metrics for the recommendation system.
    
    Parameters:
    -----------
    recommender : SHLRecommender
        Instance of the SHL recommender
    test_queries : List[Dict[str, Any]]
        List of test queries with query text and expected assessment names
    k : int, optional
        Number of recommendations to consider (default: 3)
        
    Returns:
    --------
    Tuple[float, float]
        Mean Recall@k and MAP@k scores
    """
    all_recalls = []
    all_precisions = []
    
    for test_item in test_queries:
        query = test_item.get('query', '')
        expected_assessments = test_item.get('expected_assessments', [])
        
        if not query or not expected_assessments:
            continue
            
        # Get recommendations
        recommendations = recommender.recommend(query, num_recommendations=k)
        recommended_names = [rec.get('name', '') for rec in recommendations]
        
        # Calculate Recall@k
        relevant_recommended = set(recommended_names) & set(expected_assessments)
        recall = len(relevant_recommended) / len(expected_assessments) if expected_assessments else 0
        all_recalls.append(recall)
        
        # Calculate Average Precision for MAP
        avg_precision = 0
        relevant_count = 0
        
        for j, rec_name in enumerate(recommended_names):
            if rec_name in expected_assessments:
                relevant_count += 1
                precision_at_j = relevant_count / (j + 1)
                avg_precision += precision_at_j
        
        avg_precision = avg_precision / len(expected_assessments) if expected_assessments else 0
        all_precisions.append(avg_precision)
    
    # Calculate Mean Recall@k and MAP@k
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    map_k = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    
    return mean_recall, map_k

def print_detailed_evaluation(recommender: SHLRecommender, test_queries: List[Dict[str, Any]], 
                             k: int = 3) -> None:
    """
    Print detailed evaluation results for each test query.
    
    Parameters:
    -----------
    recommender : SHLRecommender
        Instance of the SHL recommender
    test_queries : List[Dict[str, Any]]
        List of test queries with query text and expected assessment names
    k : int, optional
        Number of recommendations to consider (default: 3)
    """
    print(f"\n{'='*80}")
    print(f"Detailed Evaluation Results (k={k})")
    print(f"{'='*80}")
    
    all_recalls = []
    all_precisions = []
    
    for i, test_item in enumerate(test_queries):
        query = test_item.get('query', '')
        expected_assessments = test_item.get('expected_assessments', [])
        
        if not query or not expected_assessments:
            continue
            
        print(f"\nTest Query {i+1}: {query[:50]}..." if len(query) > 50 else f"\nTest Query {i+1}: {query}")
        print(f"Expected Assessment Names: {expected_assessments}")
        
        # Get recommendations
        recommendations = recommender.recommend(query, num_recommendations=k)
        recommended_names = [rec.get('name', '') for rec in recommendations]
        
        print(f"Recommended Assessment Names: {recommended_names}")
        
        # Calculate Recall@k
        relevant_recommended = set(recommended_names) & set(expected_assessments)
        recall = len(relevant_recommended) / len(expected_assessments) if expected_assessments else 0
        all_recalls.append(recall)
        
        # Calculate Average Precision for MAP
        avg_precision = 0
        relevant_count = 0
        
        for j, rec_name in enumerate(recommended_names):
            if rec_name in expected_assessments:
                relevant_count += 1
                precision_at_j = relevant_count / (j + 1)
                avg_precision += precision_at_j
                print(f"  Relevant match at position {j+1}: {rec_name}")
        
        avg_precision = avg_precision / len(expected_assessments) if expected_assessments else 0
        all_precisions.append(avg_precision)
        
        print(f"Matches found: {len(relevant_recommended)} out of {len(expected_assessments)}")
        print(f"Recall@{k}: {recall:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
    
    # Calculate Mean Recall@k and MAP@k
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    map_k = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    
    print(f"\n{'='*80}")
    print(f"Overall Metrics (k={k}):")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"MAP@{k}: {map_k:.4f}")
    print(f"{'='*80}")

def main():
    """Main function to run evaluation from command line."""
    parser = argparse.ArgumentParser(description='Evaluate SHL Assessment Recommendation Engine')
    parser.add_argument('--data', type=str, required=True, help='Path to SHL assessments JSON file')
    parser.add_argument('--test', type=str, required=True, help='Path to test queries JSON file')
    parser.add_argument('--k', type=int, default=3, help='Number of recommendations to consider (default: 3)')
    parser.add_argument('--detailed', action='store_true', help='Print detailed evaluation results')
    
    args = parser.parse_args()
    
    # Initialize recommender
    recommender = SHLRecommender(data_file=args.data)
    
    # Load test queries
    test_queries = load_test_queries(args.test)
    
    if not test_queries:
        print("No valid test queries found. Exiting.")
        return
    
    if args.detailed:
        print_detailed_evaluation(recommender, test_queries, k=args.k)
    else:
        mean_recall, map_k = calculate_metrics(recommender, test_queries, k=args.k)
        print(f"\nEvaluation Results (k={args.k}):")
        print(f"Mean Recall@{args.k}: {mean_recall:.4f}")
        print(f"MAP@{args.k}: {map_k:.4f}")

if __name__ == "__main__":
    main()