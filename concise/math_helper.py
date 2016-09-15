# helper functions
# mean-squared error
import numpy as np

def mse(x, y):
    return ((x - y) ** 2).mean(axis=None)

# exponentiated root-mean-squared error
def ermse(x, y):
    return 10**np.sqrt(mse(x, y))
