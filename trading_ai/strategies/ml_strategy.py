"""ML-based trading strategy."""
import pandas as pd
from .base import Strategy, SignalType
from ..models import PricePredictor

class MLStrategy(Strategy):
    def __init__(self, predictor: PricePredictor, buy_threshold: float = 0.6, sell_threshold: float = 0.4):
        self.predictor = predictor
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def generate_signal(self, df: pd.DataFrame) -> SignalType:
        """Generate trading signal based on ML prediction."""
        prob = self.predictor.predict(df.tail(1))[0]
        
        if prob >= self.buy_threshold:
            return SignalType.BUY
        elif prob <= self.sell_threshold:
            return SignalType.SELL
        return SignalType.HOLD
