"""ML-based price prediction."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class PricePredictor:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
    
    def _create_model(self):
        if self.model_type == "rf":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    def prepare_features(self, df: pd.DataFrame, target_days: int = 5) -> tuple:
        """Prepare features and target for training."""
        df = df.copy()
        
        # Target: 1 if price goes up in next N days, 0 otherwise
        df["target"] = (df["close"].shift(-target_days) > df["close"]).astype(int)
        
        # Feature columns (exclude OHLCV and target)
        self.feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume", "target"]]
        
        df = df.dropna()
        X = df[self.feature_cols].values
        y = df["target"].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_days: int = 5) -> dict:
        """Train the prediction model."""
        X, y = self.prepare_features(df, target_days)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        return {"train_accuracy": train_acc, "test_accuracy": test_acc}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict price direction."""
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save(self, path: str):
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_cols}, path)
    
    def load(self, path: str):
        data = joblib.load(path)
        self.model, self.scaler, self.feature_cols = data["model"], data["scaler"], data["features"]
