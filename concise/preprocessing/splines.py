"""
Pre-processor for smooth tracks
"""
import numpy as np
from concise.utils.splines import BSpline


def _trunc(x, minval=None, maxval=None):
    """Truncate vector values to have values on range [minval, maxval]
    """
    x = np.copy(x)
    if minval is not None:
        x[x < minval] = minval
    if maxval is not None:
        x[x > maxval] = maxval
    return x


class EncodeSplines(object):
    """Transformer (class) for computing the B-spline basis values.

    Pre-processing step for spline transformation (`SplineT`) layer.
    This transformer works on arrays that are either `N x D` or `N x L x D` dimensional.
    Last dimension encodes different features (D) and first dimension different examples.
    Knot placement is specific for each feature individually,
    unless `share_knots` is set `True`.
    The final result is an array with a new axis:

    - `N x D -> N x D x n_bases`
    - `N x L x D -> N x L x D x n_bases`

    # Arguments
        n_bases: int; Number of basis functions.
        degree: int; 2 for quadratic, 3 for qubic splines
        share_knots: bool; if True, the spline knots are
            shared across all the features (last-dimension)

    # Methods
        fit: Calculate the knot placement from the values ranges.
        transform: Obtain the transformed values
        fit_transform: fit and transform.
    """

    def __init__(self, n_bases=10, degree=3, share_knots=False):
        self.n_bases = n_bases
        self.degree = degree
        self.share_knots = share_knots

        self.data_min_ = None
        self.data_max_ = None

    def fit(self, x):
        """Calculate the knot placement from the values ranges.

        # Arguments
            x: numpy array, either N x D or N x L x D dimensional.
        """
        assert x.ndim > 1
        self.data_min_ = np.min(x, axis=tuple(range(x.ndim - 1)))
        self.data_max_ = np.max(x, axis=tuple(range(x.ndim - 1)))

        if self.share_knots:
            self.data_min_[:] = np.min(self.data_min_)
            self.data_max_[:] = np.max(self.data_max_)

    def transform(self, x, warn=True):
        """Obtain the transformed values
        """
        # 1. split across last dimension
        # 2. re-use ranges
        # 3. Merge
        array_list = [encodeSplines(x[..., i].reshape((-1, 1)),
                                    n_bases=self.n_bases,
                                    spline_order=self.degree,
                                    warn=warn,
                                    start=self.data_min_[i],
                                    end=self.data_max_[i]).reshape(x[..., i].shape + (self.n_bases,))
                      for i in range(x.shape[-1])]
        return np.stack(array_list, axis=-2)

    def fit_transform(self, x):
        """Fit and transform.
        """
        self.fit(x)
        return self.transform(x)


# TODO - write it as a class
# TODO - lookup which methods do you exactly need
# TODO - Should we explicitly state? EncodeSplines1D, EncodeSplines0D, EncodeSplines2D?

# TODO - use as pre-processor function? - predict for the test set
def encodeSplines(x, n_bases=10, spline_order=3, start=None, end=None, warn=True):
    """**Deprecated**. Function version of the transformer class `EncodeSplines`.
    Get B-spline base-function expansion

    # Details
        First, the knots for B-spline basis functions are placed
        equidistantly on the [start, end] range.
        (inferred from the data if None). Next, b_n(x) value is
        is computed for each x and each n (spline-index) with
        `scipy.interpolate.splev`.

    # Arguments
        x: a numpy array of positions with 2 dimensions
        n_bases int: Number of spline bases.
        spline_order: 2 for quadratic, 3 for qubic splines
        start, end: range of values. If None, they are inferred from the data
        as minimum and maximum value.
        warn: Show warnings.

    # Returns
        `np.ndarray` of shape `(x.shape[0], x.shape[1], n_bases)`
    """

    # TODO - make it general...
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if start is None:
        start = np.nanmin(x)
    else:
        if x.min() < start:
            if warn:
                print("WARNING, x.min() < start for some elements. Truncating them to start: x[x < start] = start")
            x = _trunc(x, minval=start)
    if end is None:
        end = np.nanmax(x)
    else:
        if x.max() > end:
            if warn:
                print("WARNING, x.max() > end for some elements. Truncating them to end: x[x > end] = end")
            x = _trunc(x, maxval=end)
    bs = BSpline(start, end,
                 n_bases=n_bases,
                 spline_order=spline_order
                 )

    # concatenate x to long
    assert len(x.shape) == 2
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    x_long = x.reshape((-1,))

    x_feat = bs.predict(x_long, add_intercept=False)  # shape = (n_rows * n_cols, n_bases)

    x_final = x_feat.reshape((n_rows, n_cols, n_bases))
    return x_final
