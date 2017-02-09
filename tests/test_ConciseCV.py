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
from concise import math_helper
from concise import helper
from tests.setup_concise_load_data import load_example_data

class TestConciseCV(object):

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data()
        cls.json_file_path = "/tmp/cv_model.json"

        # fit the model
        param, X_feat, X_seq, y, id_vec = cls.data
        dc = concise.Concise(n_epochs=1, num_tasks=param["num_tasks"])
        cls.dcv = concise.ConciseCV(dc)
        cls.dcv.train(X_feat, X_seq, y, id_vec, n_cores=1, n_folds=3)
        cls.dcv_dict = cls.dcv.to_dict()

        # save to file
        cls.dcv.save(cls.json_file_path)

    def test_basic_methods(self):
        self.dcv.get_CV_accuracy()
        self.dcv.get_CV_models()
        self.dcv.get_folds()

    def test_get_CV_prediction(self):
        # these two should be within 5%
        y = self.data[3]
        mse1 = math_helper.mse(self.dcv.get_CV_prediction(), y)
        mse2 = np.array(list((self.dcv.get_CV_accuracy().values()))).mean()
        assert (mse1 - mse2) / mse1 < 0.05

    def test_importing_fom_dict(self):
        """
        Test importing from the dictionary
        """
        dcv2 = concise.ConciseCV.from_dict(self.dcv_dict)
        dcv2_dict = dcv2.to_dict()

        # compare the two dictionaries
        # dcv3_dict = concise.ConciseCV.from_dict(dcv2_dict).to_dict()
        dict_new = dcv2_dict["output"]["fold_1"]["output"]["weights"]
        dict_old = self.dcv_dict["output"]["fold_1"]["output"]["weights"]
        assert set(dict_new.keys()) == set(dict_old.keys())
        assert helper.compare_numpy_dict(dict_new,
                                         dict_old,
                                         exact=False)

    def test_import_from_file(self):
        # import from file
        dcv_file = concise.ConciseCV.load(self.json_file_path)

        # Test 2
        # compare the two dictionaries
        assert helper.compare_numpy_dict(dcv_file.to_dict()["output"]["fold_1"]["output"]["weights"],
                                         self.dcv_dict["output"]["fold_1"]["output"]["weights"],
                                         exact=False)

        assert helper.compare_numpy_dict(dcv_file.get_CV_accuracy(), self.dcv.get_CV_accuracy(),
                                         exact=True)
        assert np.array_equal(dcv_file.get_CV_prediction(), self.dcv.get_CV_prediction())

    @classmethod
    def teardown_class(cls):
        # remove file
        os.remove(cls.json_file_path)



class TestMultiTaskLearningCV(TestConciseCV):
    """
    Test multi-task learning
    """

    @classmethod
    def setup_class(cls):
        cls.data = load_example_data(num_tasks=3)
        cls.json_file_path = "/tmp/cv_model.json"

        # fit the model
        param, X_feat, X_seq, y, id_vec = cls.data
        dc = concise.Concise(n_epochs=1, num_tasks=param["num_tasks"])
        cls.dcv = concise.ConciseCV(dc)
        cls.dcv.train(X_feat, X_seq, y, id_vec, n_cores=1, n_folds=3)
        cls.dcv_dict = cls.dcv.to_dict()

        # save to file
        cls.dcv.save(cls.json_file_path)
