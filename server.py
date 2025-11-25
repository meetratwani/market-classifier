from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime
import threading
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Import from main.py instead of duplicating logic
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from main import train_and_predict

app = Flask(__name__)
CORS(app)

models_cache = {}
predictions_cache = {}
cache_lock = threading.Lock()

CATEGORIES = {
    'SDG_CLEAN_ENERGY': ['ICLN', 'TAN', 'ENPH', 'FSLR'],
    'SDG_HEALTH': ['JNJ', 'PFE', 'UNH', 'ABBV'],
    'TECH': ['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
    'FINANCE': ['JPM', 'BAC', 'GS', 'MS'],
    'CONSUMER': ['AMZN', 'WMT', 'HD', 'NKE'],
    'ENERGY_TRADITIONAL': ['XOM', 'CVX', 'COP', 'SLB']
}

def get_current_price(ticker):
    """Get current price for ticker"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    return None

def get_currency_symbol(ticker):
    """Determine currency symbol based on ticker exchange"""
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'  # Indian Rupee
    elif ticker.endswith('.L') or ticker.endswith('.TO'):
        return '£'  # British Pound (for LSE) or could be CAD for TSX
    elif ticker.endswith('.T') or ticker.endswith('.HK'):
        return '¥'  # Japanese Yen or Hong Kong Dollar
    else:
        return '$'  # Default to USD

def get_or_create_model(tickers):
    """Get cached model or run main.py pipeline to create new one"""
    ticker_key = ','.join(sorted(tickers))
    
    with cache_lock:
        if ticker_key in models_cache:
            cache_time = models_cache[ticker_key].get('timestamp', 0)
            if time.time() - cache_time < 3600:
                return models_cache[ticker_key]
        
        print(f"Running main pipeline for: {ticker_key}")
        # Use the clean function from main.py - trains separate model for each ticker
        result = train_and_predict(tickers, category='CUSTOM', train_per_ticker=True)
        
        if result:
            result['timestamp'] = time.time()
            models_cache[ticker_key] = result
        
        return result

def create_confusion_matrix_image(cm):
    """Generate confusion matrix as base64 image"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def create_roc_curve_image(pipeline_result):
    """Generate ROC curve as base64 image"""
    from sklearn.metrics import roc_curve, auc
    
    classifier = pipeline_result['classifier']
    X_test = pipeline_result['X_test']
    y_test = pipeline_result['y_test']
    
    y_proba = classifier.model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2, color='#3b82f6')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def create_feature_importance_image(pipeline_result, top_n=15):
    """Generate feature importance chart as base64 image"""
    try:
        classifier = pipeline_result.get('classifier')
        if classifier is None:
            return None
        
        feature_names = classifier.feature_names
        
        if hasattr(classifier.model, 'feature_importances_'):
            importances = classifier.model.feature_importances_
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(8, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"  Warning: Feature importance chart failed: {e}")
        return None

def create_price_history_chart(pipeline_result):
    """Generate price history chart as base64 image"""
    try:
        tickers = pipeline_result.get('tickers', [])
        if not tickers:
            # Try to get from loader
            loader = pipeline_result.get('loader')
            if loader and hasattr(loader, 'tickers'):
                tickers = loader.tickers
        
        raw_data = pipeline_result.get('raw_data')
        loader = pipeline_result.get('loader')
        
        plt.figure(figsize=(10, 6))
        
        if raw_data is not None and tickers:
            for ticker in tickers:
                try:
                    if loader and hasattr(loader, 'get_ticker_data'):
                        ticker_df = loader.get_ticker_data(ticker)
                    else:
                        # Try to extract from raw_data directly
                        if isinstance(raw_data.columns, pd.MultiIndex):
                            ticker_df = raw_data[ticker].copy()
                        else:
                            cols = [col for col in raw_data.columns if ticker in str(col)]
                            ticker_df = raw_data[cols].copy()
                            if 'Close' not in ticker_df.columns and len(ticker_df.columns) > 0:
                                # Use first column as Close
                                ticker_df['Close'] = ticker_df.iloc[:, 0]
                    
                    if not ticker_df.empty and 'Close' in ticker_df.columns:
                        plt.plot(ticker_df.index, ticker_df['Close'], label=ticker, linewidth=2)
                except Exception as e:
                    print(f"  Warning: Could not plot {ticker}: {e}")
                    continue
        else:
            # Fallback: try to plot from ticker_df if available
            ticker_df = pipeline_result.get('ticker_df')
            if ticker_df is not None and 'Close' in ticker_df.columns:
                plt.plot(ticker_df.index, ticker_df['Close'], linewidth=2)
                ticker_name = pipeline_result.get('ticker', 'Stock')
                plt.title(f'{ticker_name} Price History (Last 2 Years)')
        
        if len(plt.gca().get_lines()) == 0:
            # No data plotted, create empty chart
            plt.text(0.5, 0.5, 'No price data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title('Price History (Last 2 Years)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        if len(plt.gca().get_lines()) > 0:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"  Warning: Price history chart failed: {e}")
        return None

def create_returns_distribution_chart(pipeline_result):
    """Generate returns distribution as base64 image"""
    try:
        # Try to get data from different possible structures
        combined_df = pipeline_result.get('combined_df')
        ticker_df = pipeline_result.get('ticker_df')
        primary_ticker = pipeline_result.get('primary_ticker')
        ticker = pipeline_result.get('ticker')
        
        # Determine which dataframe and ticker to use
        if combined_df is not None and primary_ticker:
            df = combined_df
            ticker_name = primary_ticker
            return_col = f'{primary_ticker}_return_1d'
        elif ticker_df is not None:
            df = ticker_df
            ticker_name = ticker or 'Stock'
            return_col = 'return_1d'
        else:
            return None
        
        if return_col not in df.columns:
            return None
        
        returns = df[return_col].dropna()
        
        if len(returns) == 0:
            return None
        
        plt.figure(figsize=(8, 5))
        plt.hist(returns * 100, bins=50, color='#3b82f6', alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        plt.title(f'{ticker_name} Daily Returns Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"  Warning: Returns distribution chart failed: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - trains separate model for each ticker and returns predictions with estimated prices"""
    try:
        data = request.get_json()
        tickers_input = data.get('tickers', '')
        
        if isinstance(tickers_input, str):
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        else:
            tickers = [str(t).strip().upper() for t in tickers_input if str(t).strip()]
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        # Train separate model for each ticker
        pipeline_result = get_or_create_model(tickers)
        
        if not pipeline_result:
            return jsonify({'error': 'Failed to train models or insufficient data'}), 500
        
        predictions = []
        all_metrics = []
        
        # Check if we have per-ticker results
        if pipeline_result.get('train_per_ticker') and 'results' in pipeline_result:
            # Multiple tickers, each with their own model
            for ticker, result in pipeline_result['results'].items():
                pred = result['prediction']
                currency = get_currency_symbol(ticker)
                current_price = result.get('current_price') or get_current_price(ticker) or 0.0
                
                predictions.append({
                    'ticker': ticker,
                    'currentPrice': round(current_price, 2),
                    'estimatedPrice': round(pred.get('estimated_price', current_price), 2) if pred.get('estimated_price') else None,
                    'direction': pred['prediction'],  # 'UP' or 'DOWN'
                    'arrow': pred.get('arrow', '↑' if pred['prediction'] == 'UP' else '↓'),
                    'confidence': round(pred['confidence'] * 100, 1),
                    'probabilityUp': round(pred['probability_up'] * 100, 1),
                    'probabilityDown': round(pred['probability_down'] * 100, 1),
                    'currency': currency,
                    'isPositive': pred['prediction'] == 'UP',
                    'hasRealPrediction': True
                })
                
                all_metrics.append({
                    'ticker': ticker,
                    'accuracy': round(result['metrics']['test_accuracy'], 4),
                    'roc_auc': round(result['metrics']['test_roc_auc'], 4)
                })
        else:
            # Single ticker or combined model (legacy format)
            primary_ticker = pipeline_result.get('primary_ticker', tickers[0])
            main_prediction = pipeline_result['prediction']
            currency = get_currency_symbol(primary_ticker)
            current_price = get_current_price(primary_ticker) or 0.0
            
            estimated_price = main_prediction.get('estimated_price') if isinstance(main_prediction, dict) else None
            
            predictions.append({
                'ticker': primary_ticker,
                'currentPrice': round(current_price, 2),
                'estimatedPrice': round(estimated_price, 2) if estimated_price else None,
                'direction': main_prediction['prediction'] if isinstance(main_prediction, dict) else main_prediction,
                'arrow': main_prediction.get('arrow', '↑' if (main_prediction['prediction'] == 'UP' if isinstance(main_prediction, dict) else False) else '↓'),
                'confidence': round(main_prediction['confidence'] * 100, 1) if isinstance(main_prediction, dict) else 0,
                'probabilityUp': round(main_prediction['probability_up'] * 100, 1) if isinstance(main_prediction, dict) else 0,
                'probabilityDown': round(main_prediction['probability_down'] * 100, 1) if isinstance(main_prediction, dict) else 0,
                'currency': currency,
                'isPositive': (main_prediction['prediction'] == 'UP' if isinstance(main_prediction, dict) else False),
                'hasRealPrediction': True
            })
            
            all_metrics.append({
                'ticker': primary_ticker,
                'accuracy': round(pipeline_result['metrics']['test_accuracy'], 4),
                'roc_auc': round(pipeline_result['metrics']['test_roc_auc'], 4)
            })
        
        if not predictions:
            return jsonify({'error': 'No predictions could be generated'}), 500
        
        return jsonify({
            'predictions': predictions,
            'metadata': {
                'models_trained': len(predictions),
                'per_ticker_models': pipeline_result.get('train_per_ticker', False),
                'metrics': all_metrics
            }
        })
        
    except Exception as e:
        print(f"API error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations', methods=['POST'])
def get_visualizations():
    """Get all visualization charts"""
    try:
        data = request.get_json()
        tickers_input = data.get('tickers', '')
        
        if isinstance(tickers_input, str):
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        else:
            tickers = [str(t).strip().upper() for t in tickers_input if str(t).strip()]
        
        if not tickers:
            return jsonify({'error': 'No tickers provided'}), 400
        
        pipeline_result = get_or_create_model(tickers)
        
        if not pipeline_result:
            return jsonify({'error': 'Model not found or failed to train'}), 500
        
        # Handle per-ticker results (new format)
        if pipeline_result.get('train_per_ticker') and 'results' in pipeline_result:
            # Use the first ticker's results for visualizations (or primary ticker if single)
            primary_ticker = tickers[0] if len(tickers) == 1 else tickers[0]
            if primary_ticker in pipeline_result['results']:
                result = pipeline_result['results'][primary_ticker]
                metrics = result['metrics']
                
                # Create a compatible structure for visualization functions
                # Create a mock loader object
                class MockLoader:
                    def __init__(self, ticker_df, ticker_name):
                        self.ticker_df = ticker_df
                        self.ticker_name = ticker_name
                        self.tickers = [ticker_name]
                    
                    def get_ticker_data(self, ticker):
                        if ticker == self.ticker_name:
                            return self.ticker_df
                        return None
                
                mock_loader = MockLoader(result['ticker_df'], primary_ticker)
                
                viz_result = {
                    'classifier': result['classifier'],
                    'X_test': result['X_test'],
                    'y_test': result['y_test'],
                    'tickers': [primary_ticker],
                    'loader': mock_loader,
                    'raw_data': result['raw_data'],
                    'ticker_df': result['ticker_df'],  # For returns distribution
                    'ticker': primary_ticker,  # For returns distribution
                    'primary_ticker': primary_ticker,  # For returns distribution
                    'combined_df': result['ticker_df']  # Alias for compatibility
                }
                
                visualizations = {
                    'confusionMatrix': create_confusion_matrix_image(metrics['confusion_matrix']),
                    'rocCurve': create_roc_curve_image(viz_result),
                    'featureImportance': create_feature_importance_image(viz_result),
                    'priceHistory': create_price_history_chart(viz_result),
                    'returnsDistribution': create_returns_distribution_chart(viz_result),
                    'metrics': {
                        'train_accuracy': round(metrics['train_accuracy'], 4),
                        'test_accuracy': round(metrics['test_accuracy'], 4),
                        'test_f1': round(metrics['test_f1'], 4),
                        'test_roc_auc': round(metrics['test_roc_auc'], 4),
                        'confusion_matrix': metrics['confusion_matrix'].tolist()
                    }
                }
            else:
                return jsonify({'error': 'No results found for primary ticker'}), 500
        else:
            # Handle legacy format (single combined model)
            metrics = pipeline_result.get('metrics', {})
            if not metrics:
                return jsonify({'error': 'No metrics found in pipeline result'}), 500
            
            visualizations = {
                'confusionMatrix': create_confusion_matrix_image(metrics['confusion_matrix']),
                'rocCurve': create_roc_curve_image(pipeline_result),
                'featureImportance': create_feature_importance_image(pipeline_result),
                'priceHistory': create_price_history_chart(pipeline_result),
                'returnsDistribution': create_returns_distribution_chart(pipeline_result),
                'metrics': {
                    'train_accuracy': round(metrics['train_accuracy'], 4),
                    'test_accuracy': round(metrics['test_accuracy'], 4),
                    'test_f1': round(metrics['test_f1'], 4),
                    'test_roc_auc': round(metrics['test_roc_auc'], 4),
                    'confusion_matrix': metrics['confusion_matrix'].tolist()
                }
            }
        
        return jsonify(visualizations)
        
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available stock categories"""
    categories = []
    for cat, tickers in CATEGORIES.items():
        categories.append({
            'name': cat,
            'tickers': tickers,
            'description': cat.replace('_', ' ').title()
        })
    return jsonify({'categories': categories})

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cached_models': len(models_cache)
    })

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache"""
    with cache_lock:
        models_cache.clear()
    return jsonify({'status': 'cache cleared'})

if __name__ == '__main__':
    print("="*70)
    print("MARKET MOVEMENT CLASSIFIER API SERVER")
    print("="*70)
    print("API Endpoints:")
    print("  POST /api/predict              - Predict custom tickers")
    print("  POST /api/visualizations       - Get model visualizations")
    print("  GET  /api/categories           - List all categories")
    print("  GET  /api/health               - Health check")
    print("  POST /api/clear-cache          - Clear model cache")
    print("="*70)
    print("\nServer starting on http://localhost:5000")
    print("Frontend should connect to: http://localhost:5000/api")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)