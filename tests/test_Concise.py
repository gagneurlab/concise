#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_concise
----------------------------------

Tests for `concise` module.
"""
import pytest

from concise import concise
from tests.setup_concise_load_data import load_example_data
import numpy as np

class TestConciseBasic(object):

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data()
        # pass

    def test_something(self):
        assert len(self.data) == 5

    def test_no_error(self):
        # test the nice print:
        param, X_feat, X_seq, y, id_vec = self.data
        dc = concise.Concise(n_epochs=1, **param)
        dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=1)

        dc.get_weights()

        dc.get_accuracy()

        dc.print_weights()
        # dc.plot_accuracy()
        # dc.plot_pos_bias()
        y_pred = dc.predict(X_feat, X_seq)
        y_pred

    def test_no_error_pos_bias(self):
        param, X_feat, X_seq, y, id_vec = self.data
        # with positional bias
        dc = concise.Concise(n_epochs=1, n_splines=10, num_tasks=param["num_tasks"])
        dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=1)

        dc.get_weights()
        dc.get_execution_time()
        dc.get_accuracy()
        dc.print_weights()
        # dc.plot_accuracy()
        # dc.plot_pos_bias()

        dc.predict(X_feat, X_seq)

    @classmethod
    def teardown_class(cls):
        pass


class TestMultiTaskLearning(TestConciseBasic):
    """
    Test multi-task learning
    """

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(num_tasks=3)


