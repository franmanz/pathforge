import numpy as np
import pandas as pd


class BaseModel:
    """Base class for all PathForge simulation models."""

    def __init__(self, returns):
        self.returns = returns.values.astype(float)
        self.params_ = {}

    def fit(self):
        raise NotImplementedError

    def sample(self, days, n_paths):
        raise NotImplementedError
    
    