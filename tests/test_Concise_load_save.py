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

from concise import concise
from concise import helper
from tests.setup_concise_load_data import load_example_data

class TestConciseLoad(object):
    """
    Test saving/loading to file
    """
    @classmethod
    def setup_class(cls):
        cls.data = load_example_data()
        cls.json_file_path = "/tmp/model.json"

        param, X_feat, X_seq, y, id_vec = cls.data
        cls.dc = concise.Concise(n_epochs=1)
        cls.dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=1)

        cls.mydict = cls.dc.to_dict()
        cls.y_old = cls.dc.predict(X_feat, X_seq)

        # save to file
        cls.dc.save(cls.json_file_path)

    # TODO: check this function
    # Step    0 (epoch 0): loss nan, train mse: nan, validation mse: nan
    # Step  100 (epoch 0): loss nan, train mse: nan, validation mse: nan
    # - why doesn't it fit?
    def test_concise_dict_equality(self):
        param, X_feat, X_seq, y, id_vec = self.data
        dcnew = concise.Concise.from_dict(self.mydict)
        # why does this d
        y_new = dcnew.predict(X_feat, X_seq)

        # Works!!:)

        # test1
        assert np.array_equal(y_new, self.y_old)

        # test2
        assert helper.compare_numpy_dict(self.dc.get_weights(), dcnew.get_weights(), exact=False)

    def test_concise_file_equality(self):
        param, X_feat, X_seq, y, id_vec = self.data
        dcnew2 = concise.Concise.load(self.json_file_path)
        y_new2 = dcnew2.predict(X_feat, X_seq)

        # test1
        assert np.array_equal(y_new2, self.y_old)

        # test2
        assert helper.compare_numpy_dict(self.dc.get_weights(), dcnew2.get_weights(), exact=False)

    @classmethod
    def teardown_class(cls):
        # remove file
        os.remove(cls.json_file_path)


class TestConciseLoadPosBias(TestConciseLoad):
    """
    Same TestConciseLoad, but using 
    """
    @classmethod
    def setup_class(cls):
        cls.data = load_example_data()
        cls.json_file_path = "/tmp/model_pos_bias.json"

        param, X_feat, X_seq, y, id_vec = cls.data
        cls.dc = concise.Concise(n_epochs=1, n_splines=5)
        cls.dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=3)

        cls.mydict = cls.dc.to_dict()
        cls.y_old = cls.dc.predict(X_feat, X_seq)

        # save to file
        cls.dc.save(cls.json_file_path)


class TestConciseLoadMultiClass(TestConciseLoad):
    """
    Same TestConciseLoad, but using 
    """
    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(num_tasks=3)
        cls.json_file_path = "/tmp/model_pos_bias_3_tasks.json"

        param, X_feat, X_seq, y, id_vec = cls.data
        cls.dc = concise.Concise(n_epochs=1, n_splines=5, num_tasks=param["num_tasks"])
        cls.dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=3)

        cls.mydict = cls.dc.to_dict()
        cls.y_old = cls.dc.predict(X_feat, X_seq)

        # save to file
        cls.dc.save(cls.json_file_path)
