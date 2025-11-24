import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator

class FeatureEngineer:
    """Create technical indicators and features for stock prediction"""
    
    def __init__(self):
        self.feature_columns = []
    
    def create_returns(self, df, col='Close', periods=[1, 3, 5, 7, 10]):
        """Calculate returns over multiple periods"""
        for period in periods:
            df[f'return_{period}d'] = df[col].pct_change(period)
            # Cap returns at ±100% to avoid extreme values
            df[f'return_{period}d'] = df[f'return_{period}d'].clip(-1.0, 1.0)
        return df
    
    def create_moving_averages(self, df, col='Close', windows=[5, 10, 20, 50]):
        """Calculate simple moving averages"""
        for window in windows:
            if len(df) >= window:
                df[f'sma_{window}'] = df[col].rolling(window=window).mean()
                # Avoid division by zero
                df[f'price_to_sma_{window}'] = df[col] / df[f'sma_{window}'].replace(0, np.nan)
                df[f'sma_{window}_slope'] = df[f'sma_{window}'].diff(5)
        return df
    
    def create_volatility_features(self, df, col='Close', windows=[10, 20, 30]):
        """Calculate volatility metrics"""
        returns = df[col].pct_change().clip(-1.0, 1.0)
        for window in windows:
            if len(df) >= window:
                df[f'volatility_{window}d'] = returns.rolling(window=window).std()
                df[f'rolling_max_{window}d'] = df[col].rolling(window=window).max()
                df[f'rolling_min_{window}d'] = df[col].rolling(window=window).min()
                # Avoid division by zero
                denominator = df[f'rolling_min_{window}d'].replace(0, np.nan)
                df[f'price_range_{window}d'] = (df[f'rolling_max_{window}d'] - df[f'rolling_min_{window}d']) / denominator
        return df
    
    def create_momentum_indicators(self, df, close_col='Close', high_col='High', low_col='Low'):
        """Calculate RSI, MACD, and other momentum indicators"""
        if len(df) < 26:
            return df
            
        try:
            # RSI
            rsi = RSIIndicator(close=df[close_col], window=14)
            df['rsi_14'] = rsi.rsi()
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            
            # MACD
            macd = MACD(close=df[close_col])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['macd_cross'] = ((df['macd'] > df['macd_signal']).astype(int) - 
                               (df['macd'] < df['macd_signal']).astype(int))
            
            # Rate of Change (capped)
            df['roc_10'] = ((df[close_col] - df[close_col].shift(10)) / df[close_col].shift(10).replace(0, np.nan)) * 100
            df['roc_10'] = df['roc_10'].clip(-50, 50)
            
            df['roc_20'] = ((df[close_col] - df[close_col].shift(20)) / df[close_col].shift(20).replace(0, np.nan)) * 100
            df['roc_20'] = df['roc_20'].clip(-50, 50)
            
            # Momentum
            df['momentum_10'] = df[close_col] - df[close_col].shift(10)
            
        except Exception as e:
            print(f"  Warning: Some momentum indicators failed: {e}")
        
        return df
    
    def create_bollinger_bands(self, df, col='Close', window=20):
        """Calculate Bollinger Bands"""
        if len(df) < window:
            return df
            
        try:
            bb = BollingerBands(close=df[col], window=window, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            # Avoid division by zero
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid'].replace(0, np.nan)
            denominator = (df['bb_high'] - df['bb_low']).replace(0, np.nan)
            df['bb_position'] = (df[col] - df['bb_low']) / denominator
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
        except Exception as e:
            print(f"  Warning: Bollinger Bands failed: {e}")
        
        return df
    
    def create_volume_features(self, df, volume_col='Volume'):
        """Volume-based features"""
        if volume_col in df.columns and len(df) >= 20:
            df['volume_sma_20'] = df[volume_col].rolling(window=20).mean()
            # Avoid division by zero
            df['volume_ratio'] = df[volume_col] / df['volume_sma_20'].replace(0, np.nan)
            df['volume_change'] = df[volume_col].pct_change().clip(-1.0, 1.0)
            df['volume_trend'] = df['volume_sma_20'].diff(5)
            df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        return df
    
    def create_lag_features(self, df, col='Close', lags=[1, 2, 3, 5, 10]):
        """Create lagged price features"""
        for lag in lags:
            if len(df) > lag:
                df[f'close_lag_{lag}'] = df[col].shift(lag)
                df[f'return_lag_{lag}'] = df[col].pct_change().shift(lag).clip(-1.0, 1.0)
        return df
    
    def create_interaction_features(self, df, close_col='Close'):
        """Create feature interactions"""
        if 'rsi_14' in df.columns and 'volatility_20d' in df.columns:
            df['rsi_vol_interaction'] = df['rsi_14'] * df['volatility_20d']
        
        if 'return_1d' in df.columns and 'volume_ratio' in df.columns:
            df['return_volume_interaction'] = df['return_1d'] * df['volume_ratio']
        
        return df
    
    def create_all_features(self, df, ticker_prefix=''):
        """Create all features for a single ticker"""
        close_col = f'{ticker_prefix}Close' if ticker_prefix else 'Close'
        high_col = f'{ticker_prefix}High' if ticker_prefix else 'High'
        low_col = f'{ticker_prefix}Low' if ticker_prefix else 'Low'
        volume_col = f'{ticker_prefix}Volume' if ticker_prefix else 'Volume'
        
        df = self.create_returns(df, col=close_col)
        df = self.create_moving_averages(df, col=close_col)
        df = self.create_volatility_features(df, col=close_col)
        df = self.create_momentum_indicators(df, close_col, high_col, low_col)
        df = self.create_bollinger_bands(df, col=close_col)
        df = self.create_volume_features(df, volume_col)
        df = self.create_lag_features(df, col=close_col)
        df = self.create_interaction_features(df, close_col)
        
        # Final cleanup - replace inf and extreme values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def create_target_label(self, df, target_ticker, forward_days=1):
        """Create binary target WITHOUT DATA LEAKAGE"""
        close_col = f'{target_ticker}_Close'
        
        df['future_close'] = df[close_col].shift(-forward_days)
        df['target'] = (df['future_close'] > df[close_col]).astype(int)
        
        df = df[:-forward_days]
        
        return df
    
    def create_target_label_significant(self, df, target_ticker, threshold=0.02):
        """Create target for significant movements only"""
        close_col = f'{target_ticker}_Close'
        
        df['future_close'] = df[close_col].shift(-1)
        df['return'] = (df['future_close'] - df[close_col]) / df[close_col].replace(0, np.nan)
        
        # Only keep significant movements
        df['target'] = np.where(df['return'] > threshold, 1,
                       np.where(df['return'] < -threshold, 0, np.nan))
        
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
        
        return df