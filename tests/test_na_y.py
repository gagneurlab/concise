#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test when some values are non-NA
----------------------------------
"""
import pytest
import os
import numpy as np

# TODO - use keras conde


# from concise import concise
# from tests.setup_concise_load_data import load_example_data

# # replace some Y values with NA
# def data_w_na():
#     data = load_example_data(num_tasks=3)
#     param, X_feat, X_seq, y, id_vec = data
#     y[0:51, 0] = np.NaN
#     y[51:101, 1] = np.NaN
#     y[102:300, 1] = np.NaN
#     data = (param, X_feat, X_seq, y, id_vec)
#     return data

# def test_not_na():
#     param, X_feat, X_seq, y, id_vec = data_w_na()

#     c = concise.Concise(**param)
#     c.train(X_feat, X_seq, y,
#             X_feat_valid=X_feat, X_seq_valid=X_seq, y_valid=y)

#     # Not a single one can be Nan
#     assert not np.any(np.isnan(c.get_accuracy()["loss_history"]))
#     assert not np.any(np.isnan(c.get_accuracy()["train_acc_history"]))
#     assert not np.any(np.isnan(c.get_accuracy()["val_acc_history"]))
