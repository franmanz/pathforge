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
| Geometric Brownian Motion | `"gbm"` | Fast baseline, simple assumptions |
| GARCH(1,1) | `"garch"` | Realistic volatility clustering |
| Block Bootstrap | `"bootstrap"` | Non-parametric, no distributional assumptions |
| Merton Jump Diffusion | `"jump_diffusion"` | Capturing sudden crashes and spikes |

### Which model should I use?

- **GBM** — good sanity check, fast, but underestimates tail risk
- **GARCH** — best for most use cases, captures the volatility clustering seen in real markets
- **Bootstrap** — most honest for strategy testing, resamples real historical behaviour directly
- **Jump Diffusion** — best when your data contains sudden large moves you want to preserve

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

- [ ] Poisson jump diffusion ✅
- [ ] Intraday timeframes (1m, 5m, 15m, 1h)
- [ ] Multi-asset correlated simulation
- [ ] Regime switching model
- [ ] CLI: `pathforge simulate AAPL --days 252 --paths 500`

## Contributing

PRs and issues welcome at [github.com/franmanz/pathforge](https://github.com/franmanz/pathforge).

## License

MIT © 2026 franmanz