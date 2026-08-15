import json
from pathlib import Path
import numpy as np

class TargetScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y_train: np.ndarray):
        if len(y_train) == 0:
            raise ValueError("Cannot fit scaler on empty array")
        self.mean = float(np.mean(y_train))
        self.std = float(np.std(y_train))
        if self.std == 0:
            self.std = 1.0  # Prevent division by zero if targets are constant

    def transform(self, y: np.ndarray | float) -> np.ndarray | float:
        return (y - self.mean) / self.std

    def inverse_transform(self, y_norm: np.ndarray | float) -> np.ndarray | float:
        return (y_norm * self.std) + self.mean

    def save(self, path: Path | str):
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f)

    @classmethod
    def load(cls, path: Path | str) -> "TargetScaler":
        with open(path, "r") as f:
            data = json.load(f)
        scaler = cls()
        scaler.mean = data["mean"]
        scaler.std = data["std"]
        return scaler
