# helper functions
# mean-squared error
import numpy as np

# ignore NA-values
def mse(x, y):
    # equivalent to TF representation
    y_diff = np.where(np.isnan(y) | np.isnan(x), 0, x - y)
    return ((y_diff) ** 2).mean(axis=None)

# exponentiated root-mean-squared error
def ermse(x, y):
    return 10**np.sqrt(mse(x, y))
