import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, data: np.ndarray) -> None:
        self.min_ = np.min(data, axis=0)
        self.max_ = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        range_ = np.where(self.max_ - self.min_ == 0, 1, self.max_ - self.min_)
        return (data - self.min_) / range_


class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data: np.ndarray) -> None:
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0, ddof=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        std_ = np.where(self.std_ == 0, 1, self.std_)
        return (data - self.mean_) / std_
