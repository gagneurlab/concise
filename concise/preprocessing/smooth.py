"""
Pre-processor for smooth tracks
"""
from ..splines import BSpline


def encodeSplines(x, n_bases=10, spline_order=3):
    """

    Arguments:
        x: a numpy array of positions with 2 dimensions
        n_splines int: Number of splines used for the positional bias.
        spline_order: 2 for quadratic, 3 for qubic splines
    """

    start = x.min()
    end = x.max()
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
