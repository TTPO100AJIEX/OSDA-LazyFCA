#!/usr/bin/env python3
"""
Script for selecting thresholds for metrics in LazyFCA classifier.

Features:
    - Automatic resumption: Results are saved after each metric, allowing safe interruption
    - If interrupted, simply re-run with the same arguments to continue from where it stopped

Usage:
    python run_thresholds.py <dataset_path> [--output_dir <dir>] [--test_size <float>]

Examples:
    # Run full grid search for base parameters, then test metric thresholds
    python run_thresholds.py ../datasets/churn.csv --output_dir results --test_size 0.1
    
    # Provide known best parameters to skip grid search and go straight to metric thresholds
    python run_thresholds.py ../datasets/churn.csv \
        --pos_supporters_covered 5 --neg_supporters_covered 10 \
        --pos_supporter_opposer_ratio 0.36 --neg_supporter_opposer_ratio 4.0
    
    # Test with custom number of thresholds
    python run_thresholds.py ../datasets/churn.csv --num_thresholds 20
    
    # Test with multiple pos_weight values
    python run_thresholds.py ../datasets/churn.csv --pos_weights 0.5 1.0 2.0 5.0
    
    # Test with a range of pos_weight values
    python run_thresholds.py ../datasets/churn.csv --pos_weight_range 0.1,5.0,10
    
    # Resume interrupted run (automatically skips completed metrics)
    python run_thresholds.py ../datasets/churn.csv  # Same command as before
"""

import sys
import os
import argparse
import itertools
from pathlib import Path

import numpy
import pandas
import tqdm
import tqdm.contrib.itertools
import sklearn.compose
import sklearn.preprocessing
import sklearn.model_selection

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lazyfca import LazyFCA
from lazyfca.classifier import METRIC_NAME_MAPPING
from utils.estimate_quality import estimate_quality


# List of metrics to test (excluding supporters_covered, supporter_opposer_ratio, and pos_weight)
METRICS = [
    'opposers_covered',
    'support',
    'error_rate',
    'precision',
    'lift',
    'wracc',
    'balanced_precision_proxy',
    'youdens_j',
    'matthews_correlation',
    'information_gain',
    'gini_gain',
    'log_odds_ratio',
    'chi_squared',
    'g_test',
    'interval_tightness',
    'description_volume',
    'simplicity_prior',
    'query_binary_similarity',
    'query_numeric_similarity',
    'query_similarity',
    'query_weighted_precision',
    'query_weighted_wracc',
    'stability',
    'robustness',
    'delta_stability',
]


def load_and_preprocess_dataset(dataset_path, test_size=0.1, random_state=42):
    """
    Load and preprocess a dataset.
    
    Args:
        dataset_path: Path to the CSV dataset
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed train/test splits
    """
    print(f"Loading dataset from {dataset_path}")
    data = pandas.read_csv(dataset_path)
    
    # Separate features and target
    X = data.drop(columns=['Class'])
    y = data['Class'].to_numpy()
        
    # Split data
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()
    categorical_cols = list(set(X_train.columns) - set(numeric_cols))
    
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Create preprocessing pipeline
    ct = sklearn.compose.ColumnTransformer(
        transformers = [
            ("numeric", 'passthrough', numeric_cols),
            ("categorical", sklearn.preprocessing.OneHotEncoder(dtype = 'bool'), categorical_cols)
        ]
    )
    
    # Transform data
    X_train = pandas.DataFrame(ct.fit_transform(X_train), columns=ct.get_feature_names_out())
    X_test = pandas.DataFrame(ct.transform(X_test), columns=ct.get_feature_names_out())
    
    # Ensure categorical features are boolean
    categorical_features = [f for f in ct.get_feature_names_out() if f.startswith("categorical__")]
    if categorical_features:
        X_train[categorical_features] = X_train[categorical_features].astype(bool)
        X_test[categorical_features] = X_test[categorical_features].astype(bool)
    
    y_train = pandas.Series(y_train)
    y_test = pandas.Series(y_test)
    
    print(f"Training set: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution - train: {y_train.value_counts().to_dict()}, test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def find_best_base_params(X_train, X_test, y_train, y_test, output_path):
    """
    Find best base parameters using grid search over supporters_covered and supporter_opposer_ratio.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        output_path: Path to save results CSV
        
    Returns:
        best_params: Dictionary with best parameters
    """
    print("\n" + "="*80)
    print("PHASE 1: Finding best base parameters")
    print("="*80)
    
    # Define parameter grid
    pos_supporters_covered = [1, 2, 3, 5, 10]
    neg_supporters_covered = [1, 2, 3, 5, 10]
    pos_supporter_opposer_ratio = [0.25, 0.5, 1.0, 2.0, 4.0]
    neg_supporter_opposer_ratio = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    param_combinations = list(itertools.product(
        pos_supporters_covered,
        neg_supporters_covered,
        pos_supporter_opposer_ratio,
        neg_supporter_opposer_ratio
    ))
    
    print(f"Testing {len(param_combinations)} parameter combinations")
    
    results = []
    best_f1 = -1
    best_params = None
    
    for pos_sup, neg_sup, pos_rat, neg_rat in tqdm.tqdm(param_combinations, desc="Grid search"):
        try:
            classifier = LazyFCA(
                pos_params=LazyFCA.Params(
                    supporters_covered=pos_sup,
                    supporter_opposer_ratio=pos_rat,
                ),
                neg_params=LazyFCA.Params(
                    supporters_covered=neg_sup,
                    supporter_opposer_ratio=neg_rat,
                ),
                pos_weight=1.0
            )
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            metrics = estimate_quality(y_pred, y_test)
            
            result = {
                'pos_supporters_covered': pos_sup,
                'neg_supporters_covered': neg_sup,
                'pos_supporter_opposer_ratio': pos_rat,
                'neg_supporter_opposer_ratio': neg_rat,
                **metrics
            }
            results.append(result)
            
            # Track best F1 score
            if metrics['F1-score'] > best_f1:
                best_f1 = metrics['F1-score']
                best_params = result
                
        except Exception as e:
            print(f"\nError with params (pos_sup={pos_sup}, neg_sup={neg_sup}, pos_rat={pos_rat}, neg_rat={neg_rat}): {e}")
            continue
    
    # Save results
    df = pandas.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nBase parameter search results saved to {output_path}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Best parameters: pos_sup={best_params['pos_supporters_covered']}, "
          f"neg_sup={best_params['neg_supporters_covered']}, "
          f"pos_rat={best_params['pos_supporter_opposer_ratio']:.2f}, "
          f"neg_rat={best_params['neg_supporter_opposer_ratio']:.2f}")
    
    return best_params


def test_metric_thresholds(X_train, X_test, y_train, y_test, base_params, output_file, pos_weights=None, num_thresholds=15):
    """
    Test different threshold values for each metric.
    
    This function supports resuming interrupted runs:
    - Results are saved after each metric completes
    - If output_file exists, already-completed metrics are skipped
    - Allows safe interruption and resumption of long-running experiments
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        base_params: Best base parameters found in phase 1
        output_file: Path to save all metric threshold results (single CSV)
        pos_weights: List of positive weights to test (default: [1.0])
        num_thresholds: Number of threshold values to test for each metric (default: 15)
    """
    if pos_weights is None:
        pos_weights = [1.0]
    print("\n" + "="*80)
    print("PHASE 2: Testing metric thresholds")
    print("="*80)
    
    # Check if results file exists and load existing results
    existing_results = []
    completed_metrics = set()
    if os.path.exists(output_file):
        print(f"\nFound existing results file: {output_file}")
        existing_df = pandas.read_csv(output_file)
        existing_results = existing_df.to_dict('records')
        completed_metrics = set(existing_df['metric_name'].unique())
        print(f"Loaded {len(existing_results)} existing results for {len(completed_metrics)} metrics: {sorted(completed_metrics)}")
    else:
        print(f"\nNo existing results file found. Will create new file: {output_file}")
    
    # Fit classifier with best base parameters to get all explanations
    print(f"Fitting classifier with best parameters to extract explanations...")
    classifier = LazyFCA(
        pos_params=LazyFCA.Params(
            supporters_covered=base_params['pos_supporters_covered'],
            supporter_opposer_ratio=base_params['pos_supporter_opposer_ratio'],
        ),
        neg_params=LazyFCA.Params(
            supporters_covered=base_params['neg_supporters_covered'],
            supporter_opposer_ratio=base_params['neg_supporter_opposer_ratio'],
        ),
        pos_weight=1.0
    )
    classifier.fit(X_train, y_train)
    
    # Get all explanations
    print("Extracting explanations from test set...")
    all_explanations = classifier.explain(X_test)
    print(f"Extracted {len(all_explanations)} explanations")
    
    # Combine all explanation scores into a single DataFrame
    print("Collecting explanation scores...")
    scores = pandas.concat(
        [explanation.display() for explanation in tqdm.tqdm(all_explanations, desc="Processing explanations")],
        axis=0
    ).drop_duplicates()
    print(f"Total unique explanations: {len(scores)}")
    
    # Collect all results across all metrics
    all_results = existing_results.copy()
    
    # Test each metric
    for metric in METRICS:
        print(f"\nProcessing metric: '{metric}'")
        
        # Skip if metric already completed
        if metric in completed_metrics:
            print(f"  Skipping '{metric}' - already completed in existing results file")
            continue
        
        score_column_name = METRIC_NAME_MAPPING.get(metric, metric)
        
        if score_column_name not in scores.columns:
            print(f"  Warning: Column '{score_column_name}' not found in scores DataFrame. Skipping.")
            continue
        
        # Get positive and negative explanation scores for this metric
        positive_scores = scores[scores["Type"] == "POSITIVE"][score_column_name]
        negative_scores = scores[scores["Type"] == "NEGATIVE"][score_column_name]
        
        # Handle infinite values
        positive_scores = positive_scores.replace([numpy.inf, -numpy.inf], numpy.nan).dropna()
        negative_scores = negative_scores.replace([numpy.inf, -numpy.inf], numpy.nan).dropna()
        
        if len(positive_scores) == 0 or len(negative_scores) == 0:
            print(f"  Warning: No valid scores found for metric '{metric}'. Skipping.")
            continue
        
        # Create threshold values for positive and negative
        positive_thresholds = numpy.linspace(positive_scores.min(), positive_scores.max(), num_thresholds)
        negative_thresholds = numpy.linspace(negative_scores.min(), negative_scores.max(), num_thresholds)
        
        print(f"  Positive thresholds: [{positive_thresholds[0]:.4f}, {positive_thresholds[-1]:.4f}]")
        print(f"  Negative thresholds: [{negative_thresholds[0]:.4f}, {negative_thresholds[-1]:.4f}]")
        print(f"  Positive weights: {pos_weights}")
        print(f"  Testing {num_thresholds * num_thresholds * len(pos_weights)} threshold combinations")
        
        # Test all combinations
        for pos_threshold, neg_threshold, pos_weight in tqdm.contrib.itertools.product(
            positive_thresholds, negative_thresholds, pos_weights, desc=f"  {metric}"
        ):
            try:
                # Count classifiers passing thresholds
                # For metrics that should be minimized, use <= threshold; for maximized, use >= threshold
                minimized_metrics = {'opposers_covered', 'error_rate', 'description_volume'}
                
                if metric in minimized_metrics:
                    num_pos_classifiers_passing = (positive_scores <= pos_threshold).sum()
                    num_neg_classifiers_passing = (negative_scores <= neg_threshold).sum()
                else:
                    num_pos_classifiers_passing = (positive_scores >= pos_threshold).sum()
                    num_neg_classifiers_passing = (negative_scores >= neg_threshold).sum()
                
                test_classifier = LazyFCA(
                    pos_params=LazyFCA.Params.from_dict({score_column_name: pos_threshold}),
                    neg_params=LazyFCA.Params.from_dict({score_column_name: neg_threshold}),
                    pos_weight=pos_weight
                )
                
                # Classify using explanations
                num_classifications = test_classifier.classify_explanations(
                    all_explanations,
                    trust=False,
                    probs=False
                )
                
                # Convert to probabilities
                total = num_classifications.sum(axis=1, keepdims=True)
                y_pred_proba = numpy.divide(num_classifications, numpy.maximum(1e-18, total))
                
                # Estimate quality
                metrics_result = estimate_quality(y_pred_proba, y_test)
                
                all_results.append({
                    "metric_name": metric,
                    "pos_threshold": pos_threshold,
                    "neg_threshold": neg_threshold,
                    "pos_weight": pos_weight,
                    "num_pos_classifiers_passing": num_pos_classifiers_passing,
                    "num_neg_classifiers_passing": num_neg_classifiers_passing,
                    "avg_positive_classifications": num_classifications[:, 1].mean(),
                    "avg_negative_classifications": num_classifications[:, 0].mean(),
                    **metrics_result,
                })
                
            except Exception as e:
                print(f"\n  Error with thresholds (pos={pos_threshold:.4f}, neg={neg_threshold:.4f}, pos_weight={pos_weight:.2f}): {e}")
                continue
        
        # Save intermediate results after each metric completes
        if all_results:
            df_all = pandas.DataFrame(all_results)
            df_all.to_csv(output_file, index=False)
            num_new_results = len(all_results) - len(existing_results)
            print(f"  Saved intermediate results: {num_new_results} new results for '{metric}' (total: {len(all_results)} results)")
    
    # Final summary
    print(f"\n{'='*80}")
    if all_results:
        df_all = pandas.DataFrame(all_results)
        num_completed = len(df_all['metric_name'].unique())
        num_new = num_completed - len(completed_metrics)
        print(f"COMPLETE: {len(all_results)} total results for {num_completed} metrics in {output_file}")
        if num_new > 0:
            print(f"  - {num_new} new metrics processed in this run")
        if len(completed_metrics) > 0:
            print(f"  - {len(completed_metrics)} metrics loaded from previous run")
        print(f"{'='*80}")
        return df_all
    else:
        print(f"No results to save")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Select thresholds for LazyFCA metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full grid search for base parameters, then test metric thresholds
  python run_thresholds.py ../datasets/churn.csv
  python run_thresholds.py data.csv --output_dir my_results --test_size 0.2
  
  # Skip grid search by providing known best parameters
  python run_thresholds.py data.csv --pos_supporters_covered 5 --neg_supporters_covered 10 --pos_supporter_opposer_ratio 0.36 --neg_supporter_opposer_ratio 4.0
  
  # Test with custom number of thresholds
  python run_thresholds.py data.csv --num_thresholds 20
  
  # Test with multiple pos_weight values
  python run_thresholds.py data.csv --pos_weights 0.5 1.0 2.0 5.0
  
  # Test with a range of pos_weight values
  python run_thresholds.py data.csv --pos_weight_range 0.1,5.0,10
        """
    )
    parser.add_argument('dataset_path', type=str, help='Path to the input CSV dataset')
    parser.add_argument('--output_dir', type=str, default='threshold_results',
                        help='Directory to save output CSV files (default: threshold_results)')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Fraction of data to use for testing (default: 0.1)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    # Optional base parameters - if all are provided, skip Phase 1
    parser.add_argument('--pos_supporters_covered', type=int, default=None,
                        help='Positive classifier supporters_covered parameter')
    parser.add_argument('--neg_supporters_covered', type=int, default=None,
                        help='Negative classifier supporters_covered parameter')
    parser.add_argument('--pos_supporter_opposer_ratio', type=float, default=None,
                        help='Positive classifier supporter_opposer_ratio parameter')
    parser.add_argument('--neg_supporter_opposer_ratio', type=float, default=None,
                        help='Negative classifier supporter_opposer_ratio parameter')
    parser.add_argument('--pos_weights', type=float, nargs='+', default=None,
                        help='List of positive weight values to test (e.g., --pos_weights 0.5 1.0 2.0). Default: [1.0]')
    parser.add_argument('--pos_weight_range', type=str, default=None,
                        help='Range of positive weights as "start,end,num" (e.g., "0.1,5.0,10" for 10 values from 0.1 to 5.0)')
    parser.add_argument('--num_thresholds', type=int, default=15,
                        help='Number of threshold values to test for each metric (default: 15)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Extract dataset name from path (without extension)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    print(f"Dataset name: {dataset_name}")
    
    # Load and preprocess dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
        args.dataset_path,
        test_size=args.test_size,
        random_state=args.random_seed
    )
    
    # Check if all base parameters are provided
    base_params_provided = all([
        args.pos_supporters_covered is not None,
        args.neg_supporters_covered is not None,
        args.pos_supporter_opposer_ratio is not None,
        args.neg_supporter_opposer_ratio is not None,
    ])
    
    if base_params_provided:
        # Use provided parameters, skip Phase 1
        print("\n" + "="*80)
        print("Using provided base parameters (skipping Phase 1)")
        print("="*80)
        best_params = {
            'pos_supporters_covered': args.pos_supporters_covered,
            'neg_supporters_covered': args.neg_supporters_covered,
            'pos_supporter_opposer_ratio': args.pos_supporter_opposer_ratio,
            'neg_supporter_opposer_ratio': args.neg_supporter_opposer_ratio,
        }
        print(f"Parameters: pos_sup={best_params['pos_supporters_covered']}, "
              f"neg_sup={best_params['neg_supporters_covered']}, "
              f"pos_rat={best_params['pos_supporter_opposer_ratio']:.2f}, "
              f"neg_rat={best_params['neg_supporter_opposer_ratio']:.2f}")
    else:
        # Phase 1: Find best base parameters
        if any([
            args.pos_supporters_covered is not None,
            args.neg_supporters_covered is not None,
            args.pos_supporter_opposer_ratio is not None,
            args.neg_supporter_opposer_ratio is not None,
        ]):
            print("\nWarning: Some base parameters provided but not all. Running full grid search.")
        
        base_params_file = os.path.join(args.output_dir, f"{dataset_name}_base_parameters.csv")
        best_params = find_best_base_params(X_train, X_test, y_train, y_test, base_params_file)
    
    # Phase 2: Test metric thresholds
    metrics_output_file = os.path.join(args.output_dir, f"{dataset_name}_metric_thresholds.csv")
    
    # Parse pos_weights
    if args.pos_weight_range is not None:
        # Parse range format: "start,end,num"
        try:
            start, end, num = args.pos_weight_range.split(',')
            pos_weights = numpy.linspace(float(start), float(end), int(num)).tolist()
            print(f"\nUsing pos_weight range: {pos_weights}")
        except ValueError as e:
            raise ValueError(f"Invalid pos_weight_range format. Expected 'start,end,num', got '{args.pos_weight_range}'") from e
    elif args.pos_weights is not None:
        pos_weights = args.pos_weights
        print(f"\nUsing pos_weights: {pos_weights}")
    else:
        pos_weights = [1.0]
        print(f"\nUsing default pos_weights: {pos_weights}")
    
    test_metric_thresholds(
        X_train, X_test, y_train, y_test, best_params, metrics_output_file,
        pos_weights=pos_weights, num_thresholds=args.num_thresholds
    )
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    if not base_params_provided:
        print(f"  - Base parameters: {base_params_file}")
    print(f"  - Metric thresholds: {metrics_output_file}")


if __name__ == "__main__":
    main()
