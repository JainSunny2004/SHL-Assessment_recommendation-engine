"""
run_evaluation.py - Script to evaluate the SHL recommendation engine.

This script runs the evaluation on the test_queries1.json dataset and reports
metrics for different k values.
"""

import json
import argparse
from recommender import SHLRecommender
from evaluate import calculate_metrics, print_detailed_evaluation

def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate SHL Assessment Recommendation Engine')
    parser.add_argument('--data', type=str, default='shl_assessments.json', 
                        help='Path to SHL assessments JSON file')
    parser.add_argument('--test', type=str, default='test_queries1.json', 
                        help='Path to test queries JSON file')
    parser.add_argument('--detailed', action='store_true', help='Print detailed evaluation results')
    
    args = parser.parse_args()
    
    # Load test queries
    try:
        with open(args.test, 'r', encoding='utf-8') as f:
            test_queries = json.load(f)
        print(f"Loaded {len(test_queries)} test queries from {args.test}")
    except Exception as e:
        print(f"Error loading test queries: {str(e)}")
        return
    
    # Initialize recommender
    try:
        recommender = SHLRecommender(data_file=args.data)
        print(f"Initialized recommender with data from {args.data}")
    except Exception as e:
        print(f"Error initializing recommender: {str(e)}")
        return
    
    # Evaluate for different k values
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    k_values = [1, 3, 5, 10]
    results = {}
    
    for k in k_values:
        mean_recall, map_k = calculate_metrics(recommender, test_queries, k=k)
        results[k] = {
            'recall': mean_recall,
            'map': map_k
        }
        print(f"\nResults for k={k}:")
        print(f"Mean Recall@{k}: {mean_recall:.4f} ({mean_recall*100:.2f}%)")
        print(f"MAP@{k}: {map_k:.4f} ({map_k*100:.2f}%)")
    
    # Save results to file
    try:
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to evaluation_results.json")
    except Exception as e:
        print(f"\nError saving results: {str(e)}")
    
    # Print detailed evaluation for specified k if requested
    if args.detailed:
        k_for_detail = 5  # Default to k=5 for detailed results
        print("\n\nDETAILED EVALUATION RESULTS")
        print_detailed_evaluation(recommender, test_queries, k=k_for_detail)

if __name__ == "__main__":
    main()
    