import numpy as np
from pathforge.models.base import BaseModel


class BlockBootstrapModel(BaseModel):
    """
    Block Bootstrap model.

    Resamples contiguous blocks of historical returns to generate
    simulated paths. Makes no distributional assumptions — the
    simulated paths are built entirely from real historical data.
    """

    def __init__(self, returns, block_size=None):
        super().__init__(returns)
        self._block_size_override = block_size

    def fit(self):
        if self._block_size_override is not None:
            block_size = self._block_size_override
        else:
            block_size = self._estimate_block_size()

        self.params_["block_size"] = block_size

    def sample(self, days, n_paths):
        block_size = self.params_["block_size"]
        n = len(self.returns)
        simulated = np.empty((days, n_paths))

        for path in range(n_paths):
            path_returns = []
            while len(path_returns) < days:
                start = np.random.randint(0, n - block_size + 1)
                block = self.returns[start: start + block_size]
                remaining = days - len(path_returns)
                path_returns.extend(block[:remaining].tolist())
            simulated[:, path] = path_returns

        return simulated

    def _estimate_block_size(self):
        """Estimate block size from the autocorrelation of squared returns."""
        sq_returns = self.returns ** 2
        n = len(sq_returns)
        max_lag = min(50, n // 5)
        threshold = 1.96 / np.sqrt(n)

        acf = np.array([
            np.corrcoef(sq_returns[:-lag], sq_returns[lag:])[0, 1]
            for lag in range(1, max_lag + 1)
        ])

        significant_lags = np.where(np.abs(acf) > threshold)[0]

        if len(significant_lags) == 0:
            return 10
        return max(10, int(significant_lags[-1]) + 1)