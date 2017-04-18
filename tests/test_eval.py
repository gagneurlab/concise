import concise.eval_metrics as ce
import numpy as np


def test_eval():

    n = 1000
    np.random.seed(seed=42)
    y_true = np.random.normal(loc=0, scale=1.0, size=(n,))
    y = y_true + np.random.normal(scale=0.5, size=(n,))

    assert ce.mse(y_true, y) > 0.2
    assert ce.mse(y_true, y) < 0.3
    assert ce.var_explained(y_true, y) > 0.7
    assert ce.var_explained(y_true, y) < 0.8
