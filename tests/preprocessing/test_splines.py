import pytest
import numpy as np
from concise.preprocessing.splines import encodeSplines, EncodeSplines, _trunc


def test_trunc():
    x = np.arange(10)

    assert np.allclose(_trunc(x, minval=2),
                       np.array([2, 2, 2, 3, 4, 5, 6, 7, 8, 9])
                       )
    assert np.allclose(_trunc(x, maxval=6),
                       np.array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])
                       )
    assert np.allclose(_trunc(x, minval=3, maxval=6),
                       np.array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
                       )


def test_encodeSplines():
    x = np.arange(100)

    assert np.allclose(encodeSplines(x, start=-1, end=50).sum(2), 1)
    assert np.allclose(encodeSplines(x, start=-1, end=120).sum(2), 1)
    assert np.allclose(encodeSplines(x, start=10, end=120).sum(2), 1)
    assert np.allclose(encodeSplines(x).sum(2), 1)


def test_EncodeSplines_simple():
    # raises error
    with pytest.raises(Exception):
        x = np.arange(10)
        es = EncodeSplines()
        es.fit(x)

    x = np.arange(10).reshape((-1, 1))
    es = EncodeSplines()
    es.fit(x)
    assert np.allclose(es.transform(x).sum(-1), 1)


@pytest.mark.parametrize("shape", [
    ((100, 3, 2)),
    ((100, 3, 4, 2)),
    ((100, 2)),
])
def test_EncodeSplines(shape):
    x = np.random.normal(size=shape)

    es = EncodeSplines()
    es.fit(x)
    x_all = es.transform(x)
    assert x_all.shape == (shape + (10, ))

    assert np.allclose(x_all[..., 1, :].sum(-1), 1)

    assert es.data_min_[0] == x[..., 0].min()
    assert es.data_max_[1] == x[..., 1].max()


def test_EncodeSplines_comparison():
    x = np.random.normal(size=(100, 3, 2))

    x1t = encodeSplines(x[..., 0])
    x2t = encodeSplines(x[..., 1])

    es = EncodeSplines()
    es.fit(x)
    x_all = es.transform(x)

    assert np.allclose(x_all[..., 0, :], x1t)
    assert np.allclose(x_all[..., 1, :], x2t)
    assert np.allclose(x_all[..., 1, :].sum(-1), 1)

    assert es.data_min_[0] == x[..., 0].min()
    assert es.data_max_[1] == x[..., 1].max()
