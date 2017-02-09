#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_concise
----------------------------------

Tests for `concise` module.
"""
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression

from concise import concise
from tests.setup_concise_load_data import load_example_data
from concise.math_helper import mse

class TestConcisePrediction(object):

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(trim_seq_len=1, standardize_features=False)
        # cls.data2 = load_example_data(trim_seq_len=1, standardize_features=True)
        cls.data[0]["n_motifs"] = 1
        cls.data[0]["motif_length"] = 1
        cls.data[0]["step_size"] = 0.001
        cls.data[0]["early_stop_patience"] = 3

    def test_non_std(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data

        dc = concise.Concise(n_epochs=50, **param)
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
        assert mse_dc < mse_lm + 0.007

        # dc.plot_accuracy()
        # dc.plot_pos_bias()

    def test_lbfgs(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data

        param["optimizer"] = "lbfgs"
        dc = concise.Concise(n_epochs=50, **param)
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
        assert mse_dc < mse_lm + 0.007
