import numpy as np
from pathforge.models.base import BaseModel


class GBMModel(BaseModel):
    """
    Geometric Brownian Motion model.

    Fits a normal distribution to historical log-returns,
    then simulates new returns using that distribution.
    """

    def fit(self):
        log_returns = np.log1p(self.returns)
        self.params_["mu"] = float(log_returns.mean())
        self.params_["sigma"] = float(log_returns.std())

    def sample(self, days, n_paths):
        mu = self.params_["mu"]
        sigma = self.params_["sigma"]

        Z = np.random.standard_normal((days, n_paths))
        log_returns = (mu - 0.5 * sigma**2) + sigma * Z
        return np.expm1(log_returns)
    
    