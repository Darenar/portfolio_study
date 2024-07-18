import pandas as pd
import numpy as np

class AnnualizedFactorMixin:
    def __init__(self, freq: str):
        self.freq = freq
        if self.freq=='daily':
            self.factor = 252
        elif self.freq=='monthly':
            self.factor = 12
        else:
            raise NotImplementedError(f"Unknown parameters freq = {freq} has been provided to the object.")


class BaseSTDCalculator(AnnualizedFactorMixin):
    def calcualate_cov_matrix(self, returns_df: pd.DataFrame):
        raise NotImplementedError
    
    def run(self, returns_df: pd.DataFrame, weights: np.ndarray=None):
        cov_matrix = self.calculate_cov_matrix(returns_df=returns_df)
        if weights is None:
            n_assets = returns_df.shape[1]
            weights = np.array([1/n_assets] * n_assets)
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * self.factor, weights)))
