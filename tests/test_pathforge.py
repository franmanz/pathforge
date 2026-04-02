import numpy as np
import pandas as pd
import pytest
import pathforge as pf


@pytest.fixture
def price_series():
    np.random.seed(0)
    returns = np.random.normal(0.0005, 0.015, 500)
    prices = 100 * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(prices, index=dates)

def test_accepts_series(price_series):
    forge = pf.PathForge(price_series)
    assert len(forge._prices) == 500


def test_accepts_dataframe(price_series):
    df = pd.DataFrame({"close": price_series, "other": price_series * 1.1})
    forge = pf.PathForge(df)
    assert len(forge._prices) == 500


def test_simulate_before_fit_raises(price_series):
    forge = pf.PathForge(price_series)
    with pytest.raises(RuntimeError):
        forge.simulate()


def test_invalid_model_raises(price_series):
    forge = pf.PathForge(price_series)
    with pytest.raises(ValueError):
        forge.fit(model="nonexistent")


@pytest.mark.parametrize("model", ["gbm", "garch", "bootstrap", "jump_diffusion"])
def test_output_shape(price_series, model):
    forge = pf.PathForge(price_series)
    forge.fit(model=model)
    sim = forge.simulate(days=252, n_paths=10, seed=42)
    assert sim.paths.shape == (253, 10)


@pytest.mark.parametrize("model", ["gbm", "garch", "bootstrap", "jump_diffusion"])
def test_prices_always_positive(price_series, model):
    forge = pf.PathForge(price_series)
    forge.fit(model=model)
    sim = forge.simulate(days=252, n_paths=10, seed=42)
    assert np.all(sim.paths > 0)


def test_reproducibility(price_series):
    forge = pf.PathForge(price_series)
    forge.fit(model="gbm")
    sim1 = forge.simulate(days=100, n_paths=10, seed=1)
    sim2 = forge.simulate(days=100, n_paths=10, seed=1)
    np.testing.assert_array_equal(sim1.paths, sim2.paths)


def test_start_price(price_series):
    forge = pf.PathForge(price_series)
    forge.fit(model="gbm")
    sim = forge.simulate(days=10, n_paths=5, start_price=500.0, seed=0)
    assert np.all(sim.paths[0] == 500.0)


def test_to_dataframe(price_series):
    forge = pf.PathForge(price_series)
    forge.fit(model="gbm")
    sim = forge.simulate(days=50, n_paths=5, seed=0)
    df = sim.to_dataframe()
    assert df.shape == (51, 5)
    assert list(df.columns) == [f"path_{i}" for i in range(5)]

    