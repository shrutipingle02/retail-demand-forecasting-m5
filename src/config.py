"""Paths and settings used everywhere else."""

import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "m5-forecasting-accuracy"
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for d in (DATA, MODELS, REPORTS):
    d.mkdir(exist_ok=True)

# History runs to d_1941. The 28 days after that are what Kaggle scored.
LAST_DAY = 1941
HORIZON = 28

# Our own holdout: the last 28 days we have actuals for.
VAL_START = 1914
VAL_END = 1941

# Drop everything before 2013 to keep memory manageable.
FIRST_DAY = 1100

SEED = 42


def set_seed(seed=SEED):
    """Make a run repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass
