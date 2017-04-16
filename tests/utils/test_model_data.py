import numpy as np
from concise.utils import model_data
from concise.utils.model_data import (split_train_test_idx,
                                      split_KFold_idx, subset)
import pytest


def test_test_len():
    x = np.zeros((10, 1))
    y_w = np.zeros((11, 1))
    y = np.zeros((10, 1))

    model_data.test_len((x, y))
    model_data.test_len(((x, y), y))
    model_data.test_len(([x, x], y))
    model_data.test_len(({"a": x, "b": x}, y))
    with pytest.raises(Exception):
        model_data.test_len(({"a": x, "b": y_w}, y))
    with pytest.raises(Exception):
        model_data.test_len(([x, y_w], y))
    with pytest.raises(Exception):
        model_data.test_len((y_w, y))


def test_train_test_split():
    x = np.zeros((10, 1))
    y = np.zeros((10, 1))
    y[:3] = 1
    train = (x, x), y

    a = split_train_test_idx(train, valid_split=.2, random_state=1)
    assert len(a[1]) == 2
    assert np.array_equal(a[0], split_train_test_idx(train, valid_split=.2,
                                                     random_state=1)[0])
    assert np.array_equal(a[1], split_train_test_idx(train, valid_split=.2,
                                                     random_state=1)[1])
    assert len(y) == len(a[0]) + len(a[1])

    # stratified test
    for i in range(20):
        assert y[split_train_test_idx(train, stratified=True)[1]].max() == 1


def test_get_kf():
    x = np.zeros((10, 1))
    y = np.zeros((10, 2))
    y[:3] = 1

    train = (x, x), y

    f = [(train, test) for train, test in split_KFold_idx(train,
                                                          cv_n_folds=3,
                                                          stratified=False)]

    assert len(f[0][0]) + len(f[0][1]) == y.shape[0]

    with pytest.raises(Exception):
        kf = split_KFold_idx(train, cv_n_folds=3, stratified=True)

    y = np.zeros((10, 1))
    y[:3] = 1

    train = (x, x), y

    for i in range(20):
        kf = split_KFold_idx(train, cv_n_folds=3, stratified=True)
        assert y[kf.__next__()[1]].max() == 1

    y = np.zeros((10))
    y[:3] = 1

    train = (x, x), y

    for i in range(20):
        kf = split_KFold_idx(train, cv_n_folds=3, stratified=True)
        assert y[kf.__next__()[1]].max() == 1


def _test_train_test_split_by_idx():
    x = np.zeros((10, 1))
    x2 = np.zeros((10, 1, 10))
    y = np.zeros((10, 1))
    y[:3] = 1

    train_t = (x, x2), y
    train_l = [x, x2], y
    train_d = {"a": x, "b": x2}, y

    train_idx, test_idx = split_train_test_idx(train_t, stratified=True)

    # tuple
    train = subset(train_t, train_idx)
    test = subset(train_t, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], list)

    # list
    train = subset(train_l, train_idx)
    test = subset(train_l, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], list)

    # dict
    train = subset(train_d, train_idx)
    test = subset(train_d, test_idx)
    assert len(train) == 2
    assert len(train[0]) == 2
    assert train[1].shape[0] == 8
    assert len(test) == 2
    assert len(test[0]) == 2
    assert isinstance(train[0], dict)

    assert train[0].keys() == train_d[0].keys()
    assert test[0].keys() == train_d[0].keys()
