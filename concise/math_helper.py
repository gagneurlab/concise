from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
# helper functions
# mean-squared error
import numpy as np

def mse(x, y):
    return ((x - y) ** 2).mean(axis=None)

# exponentiated root-mean-squared error
def ermse(x, y):
    return 10**np.sqrt(mse(x, y))
