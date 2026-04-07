import numpy as np
import pandas as pd


class SimulationResult:
    """
    The output of a PathForge simulation.

    Attributes
    ----------
    paths : np.ndarray, shape (days+1, n_paths)
        Simulated price paths. Row 0 is the starting price.
    """

    def __init__(self, paths, historical_prices=None, model_name=""):
        self.paths = paths
        self.historical_prices = historical_prices
        self.model_name = model_name

    def to_dataframe(self):
        """Return simulated paths as a pandas DataFrame."""
        n_paths = self.paths.shape[1]
        columns = [f"path_{i}" for i in range(n_paths)]
        return pd.DataFrame(self.paths, columns=columns)

    def summary(self):
        """Print a statistical summary of the simulated paths."""
        start = self.paths[0, 0]
        final = self.paths[-1, :]
        returns = (final - start) / start

        print(f"Paths         : {self.paths.shape[1]}")
        print(f"Days          : {self.paths.shape[0] - 1}")
        print(f"Start price   : {start:.4f}")
        print(f"")
        print(f"Final price distribution:")
        print(f"  Mean        : {final.mean():.4f}")
        print(f"  Median      : {np.median(final):.4f}")
        print(f"  Std         : {final.std():.4f}")
        print(f"  5th pct     : {np.percentile(final, 5):.4f}")
        print(f"  95th pct    : {np.percentile(final, 95):.4f}")
        print(f"")
        print(f"Return distribution:")
        print(f"  Mean        : {returns.mean():.2%}")
        print(f"  Median      : {np.median(returns):.2%}")
        print(f"  5th pct     : {np.percentile(returns, 5):.2%}")
        print(f"  95th pct    : {np.percentile(returns, 95):.2%}")

    def plot(self, max_paths=50):
        """
        Plot simulated price paths.

        Parameters
        ----------
        max_paths : int
            Maximum number of paths to draw. Defaults to 50.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Install matplotlib to use .plot(): pip install matplotlib")

        n = min(max_paths, self.paths.shape[1])
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(n):
            ax.plot(self.paths[:, i], alpha=0.3, lw=0.8, color="steelblue")

        median_path = np.median(self.paths, axis=1)
        ax.plot(median_path, color="darkblue", lw=2, label="Median path")

        p5 = np.percentile(self.paths, 5, axis=1)
        p95 = np.percentile(self.paths, 95, axis=1)
        ax.fill_between(range(len(p5)), p5, p95, alpha=0.15, color="steelblue", label="5–95th percentile")

        if self.historical_prices is not None:
            hist = self.historical_prices[-126:]
            hist_normalised = hist / hist.iloc[-1] * self.paths[0, 0]
            ax.plot(
                range(-len(hist), 0),
                hist_normalised.values,
                color="red",
                lw=1.5,
                label="Historical",
                zorder=5
            )

        ax.set_title(f"PathForge Simulation, {self.model_name} ({self.paths.shape[1]} paths)")
        ax.set_xlabel("Days")
        ax.set_ylabel("Price")
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.show()

    