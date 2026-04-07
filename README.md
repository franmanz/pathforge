# 🔥 pathforge

> Simulate realistic financial markets from historical price data — for strategy testing, research, and risk analysis.

[![PyPI version](https://img.shields.io/pypi/v/pathforge.svg)](https://pypi.org/project/pathforge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/franmanz/pathforge/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

## Why pathforge?

Testing a trading strategy on a single historical price series tells you how it performed on **one specific path** the market happened to take. That's not enough. A robust strategy should work across the full range of outcomes the market could have produced.

`pathforge` learns the statistical behaviour of any asset from its historical prices and generates hundreds of realistic alternative price paths. Test your strategy across all of them and you'll know how robust it really is.

## Installation
```bash
pip install pathforge
pip install numba  # required for markov_egarch model
```

To use the built-in plot functionality:
```bash
pip install pathforge[examples]
```

## Quick Start
```python
import pathforge as pf
import yfinance as yf

# Download historical price data
ticker = yf.Ticker("AAPL")
prices = ticker.history(period="5y")["Close"]

# Create a forge and fit a model
forge = pf.PathForge(prices)
forge.fit(model="garch")

# Simulate one year of trading days across 100 paths
sim = forge.simulate(days=252, n_paths=100, seed=42)

# Explore the results
sim.summary()
sim.plot()

# Get the paths as a DataFrame for your own analysis
df = sim.to_dataframe()  # shape: (253, 100)
```

## Models

| Model | `model=` | Best for |
|---|---|---|
| Markov-switching EGARCH | `"markov_egarch"` | Research-grade: hidden regimes + volatility clustering + fat tails |
| Geometric Brownian Motion | `"gbm"` | Fast baseline, simple assumptions |
| GARCH(1,1) | `"garch"` | Realistic volatility clustering |
| Block Bootstrap | `"bootstrap"` | Non-parametric, no distributional assumptions |
| Merton Jump Diffusion | `"jump_diffusion"` | Capturing sudden crashes and spikes |

### Which model should I use?

- **GBM** — good sanity check, fast, but underestimates tail risk
- **GARCH** — best for most use cases, captures the volatility clustering seen in real markets
- **Bootstrap** — most honest for strategy testing, resamples real historical behaviour directly
- **Jump Diffusion** — best when your data contains sudden large moves you want to preserve
- **Markov-switching EGARCH** — the most sophisticated model. Identifies hidden market regimes (calm, stressed, crisis) each with its own EGARCH volatility dynamics and Student-t innovations. Captures regime persistence, volatility clustering, leverage effects, and fat tails simultaneously. Requires minimum 2 years of daily data and Numba for speed optimisation.

## Usage Notes

### Markov-switching EGARCH
The `markov_egarch` model has specific requirements and options:

- **Minimum data**: 2 years of daily prices (500+ observations recommended)
- **Fitting time**: ~1 minute on a modern machine (first call longer due to Numba JIT warmup)
- **Dependencies**: requires `numba` — `pip install numba`
```python
forge = pf.PathForge(prices)
forge.fit(
    model="markov_egarch",
    n_states=3,        # number of hidden regimes
    n_starts=3,        # random restarts for EM algorithm
    verbose=True,      # print fitting progress
    random_state=42,   # for reproducibility
    min_persistence=0.7  # minimum regime persistence (set to None to disable)
)
sim = forge.simulate(days=252, n_paths=100)
```

> **Note:** This model uses a generalised EM algorithm rather than an exact closed-form M-step. Volatility dynamics are modelled using state-specific, uncentred EGARCH filters, resulting in an approximate likelihood. This approach is designed for practical simulation and backtesting rather than exact state-space inference. See the [GitHub repository](https://github.com/franmanz/pathforge) for full technical details.

## API Reference

### `PathForge(data)`

The main class. Pass a `pd.Series` or `pd.DataFrame` of daily closing prices.

| Method | Description |
|---|---|
| `.fit(model="garch")` | Fit a simulation model to the historical data |
| `.simulate(days=252, n_paths=100, start_price=None, seed=None)` | Generate simulated price paths |

### `SimulationResult`

Returned by `.simulate()`. 

| Attribute / Method | Description |
|---|---|
| `.paths` | `np.ndarray` of shape `(days+1, n_paths)` |
| `.to_dataframe()` | Paths as a `pd.DataFrame`, one column per path |
| `.summary()` | Print statistical summary of the simulation |
| `.plot(max_paths=50)` | Plot simulated paths with historical context |

## Roadmap

- [x] Merton Jump Diffusion
- [x] Markov-switching EGARCH with Student-t innovations
- [ ] Intraday timeframes (1m, 5m, 15m, 1h)
- [ ] Multi-asset correlated simulation
- [ ] Centred EGARCH specification
- [ ] CLI: `pathforge simulate AAPL --days 252 --paths 500`

## Contributing

PRs and issues welcome at [github.com/franmanz/pathforge](https://github.com/franmanz/pathforge).

## License

MIT © 2026 franmanz