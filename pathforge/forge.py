import numpy as np
import pandas as pd
from pathforge.result import SimulationResult

## MAIN CLASS
class PathForge:
    """
    Fits a statistical model to historical price data and generates
    simulated future price paths.
    """
    def __init__(self, data):
        self._prices = self._extract_prices(data)
        self._returns = self._prices.pct_change().dropna()
        self._model = None
        self._fitted = False

    def _extract_prices(self, data):
        if isinstance(data, pd.Series):
            return data.dropna()
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0].dropna()
        raise TypeError("data must be a pandas Series or DataFrame")
    
    def fit(self, model="garch"):
        """
        Fit a simulation model to the historical price data.

        Parameters
        ----------
        model : str
            "gbm", "garch", or "bootstrap"
        """
        if model == "gbm":
            from pathforge.models.gbm import GBMModel
            self._model = GBMModel(self._returns)
        elif model == "garch":
            from pathforge.models.garch import GARCHModel
            self._model = GARCHModel(self._returns)
        elif model == "bootstrap":
            from pathforge.models.bootstrap import BlockBootstrapModel
            self._model = BlockBootstrapModel(self._returns)
        elif model == "jump_diffusion":
            from pathforge.models.jump_diffusion import JumpDiffusionModel
            self._model = JumpDiffusionModel(self._returns) 
        else:
            raise ValueError(f"Unknown model '{model}'. Choose from: gbm, garch, bootstrap")

        self._model.fit()
        self._fitted = True
        return self #allows forge.fit("garch").simulate(...) in one line etc.

    #Simulation method
    def simulate(self, days=252, n_paths=100, start_price=None, seed=None):
        """
        Generate simulated price paths.

        Parameters
        ----------
        days : int
            Number of trading days to simulate. 252 is one trading year.
        n_paths : int
            Number of independent paths to generate.
        start_price : float, optional
            Starting price. Defaults to the last observed price.
        seed : int, optional
            Random seed for reproducibility.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .simulate()")

        if seed is not None:
            np.random.seed(seed)

        if start_price is None:
            start_price = float(self._prices.iloc[-1])

        returns_matrix = self._model.sample(days=days, n_paths=n_paths)
        price_paths = self._build_price_paths(returns_matrix, start_price)

        return SimulationResult(price_paths, historical_prices=self._prices, model_name=self._model.__class__.__name__)

    #Build price paths is reverse of pct_change()
    def _build_price_paths(self, returns_matrix, start_price):
        """Convert a matrix of returns into price paths."""
        n_days, n_paths = returns_matrix.shape
        price_paths = np.empty((n_days + 1, n_paths))
        price_paths[0] = start_price
        for t in range(n_days):
            price_paths[t + 1] = price_paths[t] * (1 + returns_matrix[t])
        return price_paths
    

    

        


