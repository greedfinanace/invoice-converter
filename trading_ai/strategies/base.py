"""Base strategy class."""
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> SignalType:
        pass
