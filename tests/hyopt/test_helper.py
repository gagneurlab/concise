import numpy as np
import concise
import os
import sys
import time

from concise.utils.helper import merge_dicts
from concise.hyopt import (CompileFN, CMongoTrials,
                           _test_len, _train_test_split_by_idx,
                           _get_kf, _train_test_split, _train_and_eval_single)
import subprocess
from tests.hyopt import data, model
import pytest


def test_test_len():
    x = np.zeros((10, 1))
    y_w = np.zeros((11, 1))
    y = np.zeros((10, 1))

    _test_len((x, y))
    _test_len(((x, y), y))
    _test_len(([x, x], y))
    _test_len(({"a": x, "b": x}, y))
    with pytest.raises(Exception):
        _test_len(({"a": x, "b": y_w}, y))
    with pytest.raises(Exception):
        _test_len(([x, y_w], y))
    with pytest.raises(Exception):
        _test_len((y_w, y))


def test_train_test_split():
    x = np.zeros((10, 1))
    y = np.zeros((10, 1))
    y[:3] = 1
    train = (x, x), y

    a = _train_test_split(train, valid_split=.2, random_state=1)
    assert len(a[1]) == 2
    assert np.array_equal(a[0], _train_test_split(train, valid_split=.2,
                                                  random_state=1)[0])
    assert np.array_equal(a[1], _train_test_split(train, valid_split=.2,
                                                  random_state=1)[1])
    assert len(y) == len(a[0]) + len(a[1])

    # stratified test
    for i in range(20):
        assert y[_train_test_split(train, stratified=True)[1]].max() == 1


def test_get_kf():
    x = np.zeros((10, 1))
    y = np.zeros((10, 2))
    y[:3] = 1

    train = (x, x), y

    f = [(train, test) for train, test in _get_kf(train, cv_n_folds=3, stratified=False)]

    assert len(f[0][0]) + len(f[0][1]) == y.shape[0]

    with pytest.raises(Exception):
        kf = _get_kf(train, cv_n_folds=3, stratified=True)

    y = np.zeros((10, 1))
    y[:3] = 1

    train = (x, x), y

    for i in range(20):
        kf = _get_kf(train, cv_n_folds=3, stratified=True)
        assert y[kf.__next__()[1]].max() == 1

    y = np.zeros((10))
    y[:3] = 1

    train = (x, x), y

    for i in range(20):
        kf = _get_kf(train, cv_n_folds=3, stratified=True)
        assert y[kf.__next__()[1]].max() == 1

def _train_test_split_by_idx():
    x = np.zeros((10, 1))
    x2 = np.zeros((10, 1, 10))
    y = np.zeros((10, 1))
    y[:3] = 1

    train_t = (x, x2), y
    train_l = [x, x2], y
    train_d = {"a": x, "b": x2}, y

    train_idx, test_idx = _train_test_split(train_t, stratified=True)

    # tuple
    train, test = _train_test_split_by_idx(train_t, train_idx, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], list)

    # list
    train, test = _train_test_split_by_idx(train_l, train_idx, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], list)

    # dict
    train, test = _train_test_split_by_idx(train_d, train_idx, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], dict)

    assert train[0].keys() == train_d[0].keys()
    assert test[0].keys() == train_d[0].keys()
