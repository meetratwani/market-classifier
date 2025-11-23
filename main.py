import os
import sys
import pandas as pd
import numpy as np
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
warnings.filterwarnings('ignore')

from data_loader import CleanEnergyDataLoader
from feature_engineering import FeatureEngineer
from model_training import XGBoostMarketClassifier

def main(category='SDG_CLEAN_ENERGY', custom_tickers=None, use_significant_moves=False):
    """
    Main execution pipeline for multi-category market prediction
    
    Args:
        category: Predefined category or 'CUSTOM'
        custom_tickers: List of custom tickers (if category='CUSTOM')
        use_significant_moves: If True, predict only significant moves (>2%)
    """
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("="*70)
    print("MULTI-CATEGORY MARKET MOVEMENT CLASSIFIER")
    print("="*70)
    
    # 1. DATA LOADING
    print("\n[1/6] Loading Data...")
    
    if custom_tickers:
        loader = CleanEnergyDataLoader(category='CUSTOM', custom_tickers=custom_tickers)
    else:
        loader = CleanEnergyDataLoader(category=category)
    
    # Show SDG info if applicable
    sdg_info = loader.get_sdg_info()
    if sdg_info.get('sdg_aligned'):
        print(f"\n🌍 SDG #{sdg_info['sdg_number']}: {sdg_info['sdg_name']}")
        print(f"Impact: {sdg_info['impact']}")
    
    try:
        raw_data = loader.download_data(period='2y', interval='1d')
        
        if raw_data.empty or len(raw_data) < 50:
            raise ValueError("Insufficient data downloaded")
            
    except Exception as e:
        print(f"\n⚠️  Download failed: {e}")
        print("Please check internet connection or try different tickers.")
        return None
    
    # 2. FEATURE ENGINEERING
    print("\n[2/6] Engineering Features...")
    engineer = FeatureEngineer()
    
    all_ticker_features = []
    for ticker in loader.tickers:
        print(f"  Processing {ticker}...")
        try:
            ticker_df = loader.get_ticker_data(ticker)
            
            if ticker_df.empty or len(ticker_df) < 50:
                print(f"  ⚠️  Insufficient data for {ticker}, skipping...")
                continue
                
            ticker_df = engineer.create_all_features(ticker_df, ticker_prefix='')
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            all_ticker_features.append(ticker_df)
            
        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {e}")
            continue
    
    if not all_ticker_features:
        print("\n❌ No ticker data was successfully processed!")
        return None
    
    # Combine all tickers
    combined_df = pd.concat(all_ticker_features, axis=1)
    combined_df = combined_df.dropna()
    
    if len(combined_df) < 50:
        print(f"\n❌ Insufficient samples: {len(combined_df)} (need at least 50)")
        return None
    
    # Create target label
    primary_ticker = loader.tickers[0]
    print(f"  Creating target labels using {primary_ticker}...")
    
    if use_significant_moves:
        print("  Mode: Predicting SIGNIFICANT moves (>2%)")
        combined_df = engineer.create_target_label_significant(combined_df, target_ticker=primary_ticker, threshold=0.02)
    else:
        print("  Mode: Predicting ANY move (UP/DOWN)")
        combined_df = engineer.create_target_label(combined_df, target_ticker=primary_ticker, forward_days=1)
    
    # Save processed data
    output_file = f'data/{category.lower()}_data.csv'
    combined_df.to_csv(output_file)
    print(f"  ✓ Saved: {combined_df.shape[0]} samples, {combined_df.shape[1]-1} features")
    
    # 3. MODEL TRAINING
    print("\n[3/6] Training XGBoost Model...")
    classifier = XGBoostMarketClassifier(random_state=42)
    X_train, X_test, y_train, y_test = classifier.prepare_data(combined_df)
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    if len(X_train) < 30:
        print("\n⚠️  Warning: Small training set. Results may not be reliable.")
    
    classifier.train(X_train, y_train)
    
    # 4. EVALUATION
    print("\n[4/6] Evaluating Model...")
    metrics = classifier.evaluate(X_train, X_test, y_train, y_test)
    
    print("\n" + "="*70)
    print(f"MODEL PERFORMANCE - {loader.description}")
    print("="*70)
    print(f"Training Accuracy:   {metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"Test F1-Score:       {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # 5. VISUALIZATIONS
    print("\n[5/6] Generating Visualizations...")
    result_prefix = f'results/{category.lower()}'
    classifier.plot_confusion_matrix(save_path=f'{result_prefix}_confusion_matrix.png')
    classifier.plot_roc_curve(X_test, y_test, save_path=f'{result_prefix}_roc_curve.png')
    importance_df = classifier.plot_feature_importance(top_n=20, save_path=f'{result_prefix}_feature_importance.png')
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    # 6. NEXT-DAY PREDICTION
    print("\n[6/6] Making Next-Day Prediction...")
    exclude_cols = ['target', 'future_close'] + (['return'] if use_significant_moves else [])
    latest_features = combined_df.drop(columns=[col for col in exclude_cols if col in combined_df.columns]).iloc[[-1]]
    next_day_pred = classifier.predict_next_day(latest_features)
    
    print("\n" + "="*70)
    print(f"NEXT DAY PREDICTION - {primary_ticker}")
    print("="*70)
    print(f"Predicted Movement:  {next_day_pred['prediction']}")
    print(f"Confidence:          {next_day_pred['confidence']:.2%}")
    print(f"P(DOWN):             {next_day_pred['probability_down']:.2%}")
    print(f"P(UP):               {next_day_pred['probability_up']:.2%}")
    
    # Save model
    model_file = f'models/{category.lower()}_xgboost_model.json'
    classifier.save_model(model_file)
    
    # Save metrics to file
    metrics_file = f'results/{category.lower()}_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f"MARKET MOVEMENT CLASSIFIER - {loader.description}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Category: {category}\n")
        f.write(f"Tickers: {', '.join(loader.tickers)}\n")
        f.write(f"SDG Aligned: {'Yes' if loader.sdg_aligned else 'No'}\n")
        if sdg_info.get('sdg_aligned'):
            f.write(f"SDG #{sdg_info['sdg_number']}: {sdg_info['sdg_name']}\n")
        f.write("\n")
        f.write(f"Training Accuracy:   {metrics['train_accuracy']:.4f}\n")
        f.write(f"Test Accuracy:       {metrics['test_accuracy']:.4f}\n")
        f.write(f"Test F1-Score:       {metrics['test_f1']:.4f}\n")
        f.write(f"Test ROC-AUC:        {metrics['test_roc_auc']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'] + "\n\n")
        f.write("Next Day Prediction:\n")
        f.write(f"Ticker: {primary_ticker}\n")
        f.write(f"Movement: {next_day_pred['prediction']}\n")
        f.write(f"Confidence: {next_day_pred['confidence']:.2%}\n")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Data saved to: {output_file}")
    print(f"✓ Model saved to: {model_file}")
    print(f"✓ Metrics saved to: {metrics_file}")
    print(f"✓ Visualizations in: results/")
    
    return {
        'loader': loader,
        'classifier': classifier,
        'metrics': metrics,
        'prediction': next_day_pred
    }


def run_multiple_categories(categories=None):
    """Run analysis on multiple categories"""
    if categories is None:
        categories = ['SDG_CLEAN_ENERGY', 'TECH', 'FINANCE']
    
    print("\n" + "="*70)
    print("RUNNING MULTI-CATEGORY ANALYSIS")
    print("="*70)
    
    results = {}
    
    for category in categories:
        print(f"\n\n{'='*70}")
        print(f"CATEGORY: {category}")
        print(f"{'='*70}\n")
        
        try:
            result = main(category=category)
            if result:
                results[category] = result
        except Exception as e:
            print(f"\n❌ Failed for {category}: {e}")
            continue
    
    # Summary comparison
    if results:
        print("\n\n" + "="*70)
        print("CATEGORY COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Category':<25} {'Test Acc':<12} {'ROC-AUC':<12} {'SDG':>10}")
        print("-"*70)
        
        for cat, res in results.items():
            metrics = res['metrics']
            sdg = "✓ Yes" if res['loader'].sdg_aligned else "  No"
            print(f"{cat:<25} {metrics['test_accuracy']:>10.2%}  {metrics['test_roc_auc']:>10.3f}  {sdg:>10}")
        
        print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Movement Classifier')
    parser.add_argument('--category', type=str, default='SDG_CLEAN_ENERGY',
                      help='Category to analyze (or use --list to see options)')
    parser.add_argument('--list', action='store_true',
                      help='List all available categories')
    parser.add_argument('--multi', action='store_true',
                      help='Run on multiple categories')
    parser.add_argument('--custom', type=str, nargs='+',
                      help='Custom ticker symbols (e.g., --custom TSLA NFLX AMZN)')
    parser.add_argument('--significant', action='store_true',
                      help='Predict only significant moves (>2%%)')
    
    args = parser.parse_args()
    
    # List categories
    if args.list:
        CleanEnergyDataLoader.list_categories()
    # Multi-category analysis
    elif args.multi:
        run_multiple_categories(['SDG_CLEAN_ENERGY', 'SDG_HEALTH', 'TECH', 'FINANCE', 'CONSUMER'])
    # Custom tickers
    elif args.custom:
        print(f"Running with custom tickers: {args.custom}")
        main(category='CUSTOM', custom_tickers=args.custom, use_significant_moves=args.significant)
    # Single category
    else:
        main(category=args.category, use_significant_moves=args.significant)