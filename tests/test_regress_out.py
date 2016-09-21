#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_concise
----------------------------------

Tests for `concise` module.
"""
import pytest
import os
import numpy as np
from sklearn.linear_model import LinearRegression

from concise import concise
from concise import helper
from tests.setup_concise_load_data import load_example_data
from concise.math_helper import mse

class TestConciseNormalize(object):
    """
    Test saving/loading to file
    """
    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(standardize_features=False)

    def test_compare_with_lm(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data
        param["regress_out_feat"] = True
        dc = concise.Concise(n_epochs=10, **param)
        dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=1)

        weights = dc.get_weights()
        lm = LinearRegression()
        lm.fit(X_feat, y)
        lm.coef_
        dc_coef = weights["feature_weights"].reshape(-1)

        # # weights has to be the same as for linear regression
        # (dc_coef - lm.coef_) / lm.coef_

        # they both have to predict the same
        y_pred = dc.predict(X_feat, X_seq)
        mse_lm = mse(y, lm.predict(X_feat))
        mse_dc = mse(y, y_pred)
        print("mse_lm")
        print(mse_lm)
        print("mse_dc")
        print(mse_dc)

        assert mse_dc < mse_lm + 0.005

