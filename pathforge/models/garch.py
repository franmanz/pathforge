import numpy as np
import warnings
from pathforge.models.base import BaseModel


class GARCHModel(BaseModel):
    """
    GARCH(1,1) model.

    Captures volatility clustering — the tendency for large price
    moves to be followed by more large moves.
    """

    def fit(self):
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Install arch to use GARCH: pip install arch")

        pct_returns = self.returns * 100

        am = arch_model(pct_returns, mean="Constant", vol="GARCH", p=1, q=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = am.fit(disp="off")

        params = result.params
        self.params_["mu"] = float(params["mu"]) / 100
        self.params_["omega"] = float(params["omega"]) / 10000
        self.params_["alpha"] = float(params["alpha[1]"])
        self.params_["beta"] = float(params["beta[1]"])

    def sample(self, days, n_paths):
        mu = self.params_["mu"]
        omega = self.params_["omega"]
        alpha = self.params_["alpha"]
        beta = self.params_["beta"]

        hist_var = np.var(self.returns)
        simulated = np.empty((days, n_paths))

        for path in range(n_paths):
            sigma2 = hist_var
            for t in range(days):
                eps = np.random.normal(0, np.sqrt(sigma2))
                simulated[t, path] = mu + eps
                sigma2 = omega + alpha * eps**2 + beta * sigma2

        return simulated