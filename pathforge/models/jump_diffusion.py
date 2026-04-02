import numpy as np
from pathforge.models.base import BaseModel


class JumpDiffusionModel(BaseModel):
    """
    Merton Jump Diffusion model.

    Extends GBM by adding a Poisson jump component to capture
    sudden large price moves such as crashes or earnings surprises.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    jump_threshold : float
        Number of standard deviations beyond which a historical
        return is classified as a jump. Default is 3.
    """

    def __init__(self, returns, jump_threshold=3.0):
        super().__init__(returns)
        self.jump_threshold = jump_threshold

    def fit(self):
        # Step 1 — fit the GBM component on normal days
        std = self.returns.std()
        jump_mask = np.abs(self.returns) > self.jump_threshold * std

        normal_returns = self.returns[~jump_mask]
        jump_returns = self.returns[jump_mask]

        log_normal = np.log1p(normal_returns)
        self.params_["mu"] = float(log_normal.mean())
        self.params_["sigma"] = float(log_normal.std())

        # Step 2 — fit the jump component
        n_days = len(self.returns)
        self.params_["lambda"] = float(len(jump_returns) / n_days)
        self.params_["mu_j"] = float(jump_returns.mean()) if len(jump_returns) > 0 else 0.0
        self.params_["sigma_j"] = float(jump_returns.std()) if len(jump_returns) > 1 else 0.01

    
    def sample(self, days, n_paths):
        mu = self.params_["mu"]
        sigma = self.params_["sigma"]
        lam = self.params_["lambda"]
        mu_j = self.params_["mu_j"]
        sigma_j = self.params_["sigma_j"]

        # GBM component
        Z = np.random.standard_normal((days, n_paths))
        log_returns = (mu - 0.5 * sigma**2) + sigma * Z

        # Jump component
        jumps_occur = np.random.poisson(lam, (days, n_paths))
        jump_sizes = np.random.normal(mu_j, sigma_j, (days, n_paths))
        jump_component = jumps_occur * jump_sizes

        return np.expm1(log_returns + jump_component)
    
    