"""
Pre-processor for smooth tracks
"""
import numpy as np
from concise.utils.splines import BSpline

# TODO - use as pre-processor function? - predict for the test set
def encodeSplines(x, n_bases=10, spline_order=3, start=None, end=None):
    """Get B-spline base-function expansion

    # Details
        First, the knots for B-spline basis functions are placed
        equidistantly on the [start, end] range.
        (inferred from the data if None). Next, b_n(x) value is
        is computed for each x and each n (spline-index) with
        `scipy.interpolate.splev`.

    # Arguments
        x: a numpy array of positions with 2 dimensions
        n_splines int: Number of splines used for the positional bias.
        spline_order: 2 for quadratic, 3 for qubic splines
        start, end: range of values. If None, they are inferred from the data
        as minimum and maximum value.

    # Returns
        `np.ndarray` of shape `(x.shape[0], x.shape[1], n_bases)`
    """

    if start is None:
        start = np.nanmin(x)
    if end is None:
        end = np.nanmax(x)
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
