import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import json

class IsolationForestDetector:
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest model
        contamination: expected proportion of anomalies
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.is_fitted = False
        
    def preprocess_data(self, df):
        """Convert categorical data to numeric if needed"""
        df_processed = df.copy()
        
        # Convert categorical columns to numeric
        for col in df_processed.select_dtypes(include=['object']).columns:
            df_processed[col] = pd.factorize(df_processed[col])[0]
            
        # Fill NaN with column mean
        df_processed = df_processed.fillna(df_processed.mean())
        
        return df_processed
    
    def fit(self, df):
        """Train the model"""
        df_processed = self.preprocess_data(df)
        self.model.fit(df_processed)
        self.is_fitted = True
        self.columns = df.columns.tolist()
        
    def predict(self, df):
        """Predict anomalies (-1 = anomaly, 1 = normal)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        df_processed = self.preprocess_data(df)
        predictions = self.model.predict(df_processed)
        
        # Convert to binary (0=normal, 1=anomaly)
        anomaly_flags = [1 if p == -1 else 0 for p in predictions]
        
        # Get anomaly scores
        scores = self.model.decision_function(df_processed)
        
        results = []
        for i in range(len(df)):
            result = {
                'index': i,
                'is_anomaly': anomaly_flags[i],
                'anomaly_score': float(scores[i]),
                'original_data': df.iloc[i].to_dict(),
                'suspicious_features': []
            }
            
            # Identify which features contributed to anomaly
            if anomaly_flags[i] == 1:
                # Simple feature importance: deviation from mean
                row = df_processed.iloc[i]
                means = df_processed.mean()
                deviations = abs(row - means)
                top_features = deviations.nlargest(3).index.tolist()
                result['suspicious_features'] = top_features
                
            results.append(result)
            
        return results
    
    def save_model(self, path='isolation_forest_model.pkl'):
        import joblib
        joblib.dump(self.model, path)
        
    def load_model(self, path='isolation_forest_model.pkl'):
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True