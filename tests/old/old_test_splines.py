# test splines
import numpy as np
import pytest
# from os.path import dirname
# sys.path.append(dirname(__file__))
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

# from . import splines

# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
# import splines
from functions import splines
from functions import splines_rpy2
# from imp import reload
# reload(splines)
# reload(splines_rpy2)

testdata = [(1, 100, 10, 3, True),
            (1, 100, 10, 3, False),
            (1, 100, 10, 2, True),
            (1, 100, 10, 2, False),
            (-1, 100, 5, 2, True),
            (-1, 2, 5, 2, True),
            (-1, 100, 50, 2, True),
            (-1, 100, 50, 1, True),
            (-1, 100, 50, 5, True)]

@pytest.mark.parametrize("start, end, n_bases, spline_order, add_intercept", testdata)
def test_compare_tuple(start, end, n_bases, spline_order, add_intercept):

    X1, S1, knots1 = splines.get_gam_splines(start=start, end=end,
                                             n_bases=n_bases,
                                             spline_order=spline_order,
                                             add_intercept=add_intercept)

    X2, S2, knots2 = splines_rpy2.get_gam_splines(start=start, end=end,
                                                  n_bases=n_bases,
                                                  spline_order=spline_order,
                                                  add_intercept=add_intercept)

    assert np.abs(knots1 - knots2).max() < 1e-5
    assert np.abs(X1 - X2).max() < 1e-5
    assert np.abs(S1 - S2).max() < 1e-5

