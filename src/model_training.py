import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

class XGBoostMarketClassifier:
    """Enhanced XGBoost classifier with optimization"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, df, target_col='target', test_size=0.2):
        """Prepare train/test split with proper scaling and cleaning"""
        exclude_cols = [target_col, 'future_close', 'return'] + [col for col in df.columns if 'Date' in col or 'date' in col.lower()]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # CRITICAL: Clean inf and nan values
        print(f"  Cleaning data...")
        print(f"    Before: {X.shape}, NaN: {X.isna().sum().sum()}, Inf: {np.isinf(X).sum().sum()}")
        
        # Replace inf with nan
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill nan with forward fill, then backward fill, then 0
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Double check - clip extreme values
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                col_std = X[col].std()
                col_mean = X[col].mean()
                
                # If std is valid, clip to ±5 std from mean
                if col_std > 0 and not np.isnan(col_std) and not np.isinf(col_std):
                    lower_bound = col_mean - 5 * col_std
                    upper_bound = col_mean + 5 * col_std
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Final check - replace any remaining inf/nan
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        print(f"    After: {X.shape}, NaN: {X.isna().sum().sum()}, Inf: {np.isinf(X).sum().sum()}")
        
        self.feature_names = X.columns.tolist()
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: UP={sum(y==1)}, DOWN={sum(y==0)}")
        
        # Time-series aware split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Verify no inf before scaling
        assert not np.any(np.isinf(X_train.values)), "X_train contains inf before scaling"
        assert not np.any(np.isinf(X_test.values)), "X_test contains inf before scaling"
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, params=None):
        """Train XGBoost classifier"""
        if params is None:
            params = {
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 150,
                'objective': 'binary:logistic',
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        
        self.model = xgb.XGBClassifier(**params)
        
        print("  Training XGBoost model...")
        self.model.fit(X_train, y_train, verbose=False)
        print("  ✓ Training complete!")
        
        return self.model
    
    def train_with_tuning(self, X_train, y_train, n_iter=20):
        """Train with hyperparameter tuning"""
        from sklearn.model_selection import RandomizedSearchCV
        
        param_distributions = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [100, 150, 200, 250],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.5, 1.0, 1.5]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        print(f"  Tuning hyperparameters ({n_iter} iterations)...")
        
        random_search = RandomizedSearchCV(
            xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='roc_auc',
            cv=3,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        self.model = random_search.best_estimator_
        print(f"  ✓ Best ROC-AUC: {random_search.best_score_:.4f}")
        print(f"  Best params: {random_search.best_params_}")
        
        return self.model
    
    def train_ensemble(self, X_train, y_train):
        """Train ensemble of models"""
        print("  Training ensemble (XGB + LightGBM + RF)...")
        
        xgb_model = xgb.XGBClassifier(
            max_depth=5, learning_rate=0.1, n_estimators=150,
            random_state=self.random_state, use_label_encoder=False
        )
        
        lgb_model = lgb.LGBMClassifier(
            max_depth=5, learning_rate=0.1, n_estimators=150,
            random_state=self.random_state, verbose=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=self.random_state
        )
        
        self.model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='soft',
            weights=[2, 2, 1]
        )
        
        self.model.fit(X_train, y_train)
        print("  ✓ Ensemble trained!")
        
        return self.model
    
    def time_series_cv(self, X, y, n_splits=5):
        """Time-series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            self.model.fit(X_train_scaled, y_train_fold)
            score = self.model.score(X_test_scaled, y_test_fold)
            cv_scores.append(score)
            
            print(f"    Fold {fold}: {score:.3f}")
        
        print(f"  Mean CV: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
        
        return cv_scores
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'classification_report': classification_report(y_test, y_test_pred)
        }
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path='results/confusion_matrix.png'):
        """Plot confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix - Market Movement Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Confusion matrix saved to {save_path}")
    
    def plot_roc_curve(self, X_test, y_test, save_path='results/roc_curve.png'):
        """Plot ROC curve"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.metrics["test_roc_auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Market Movement Classifier')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ ROC curve saved to {save_path}")
    
    def plot_feature_importance(self, top_n=20, save_path='results/feature_importance.png'):
        """Plot top feature importances"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        else:
            print("  ⚠️  Cannot extract feature importances")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Feature importance plot saved to {save_path}")
        
        return importance_df
    
    def predict_next_day(self, latest_features):
        """Predict next day's market movement"""
        # Clean the features before prediction
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
        latest_features = latest_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        latest_scaled = self.scaler.transform(latest_features)
        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]
        
        return {
            'prediction': 'UP' if prediction == 1 else 'DOWN',
            'probability_down': probability[0],
            'probability_up': probability[1],
            'confidence': max(probability)
        }
    
    def save_model(self, path='models/xgboost_model.json'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"  ✓ Model saved to {path}")
    
    def load_model(self, path='models/xgboost_model.json'):
        """Load trained model"""
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"  ✓ Model loaded from {path}")