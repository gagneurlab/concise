# get the splines
import numpy as np
import scipy.interpolate as si
# TODO - BSpline.predict() -> allow x to be of any shape. return.shape = in.shape + (n_bases)

# MAYBE TODO - implement si.splev using keras.backend.
#               - That way you don't have to hash the X_spline in memory.


class BSpline():
    """Class for computing the B-spline funcions b_i(x) and
    constructing the penality matrix S.

    # Arguments
        start: float or int; start of the region
        end: float or int; end of the region
        n_bases: int; number of spline bases
        spline_order: int; spline order

    # Methods
        - **getS(add_intercept=False)** - Get the penalty matrix S
              - Arguments
                     - **add_intercept**: bool. If true, intercept column is added to the returned matrix.
              - Returns
                     - `np.array`, of shape `(n_bases + add_intercept, n_bases + add_intercept)`
        - **predict(x, add_intercept=False)** - For some x, predict the bn(x) for each base
              - Arguments
                     - **x**: np.array; Vector of dimension 1
                     - **add_intercept**: bool; If True, intercept column is added to the to the final array
              - Returns
                     - `np.array`, of shape `(len(x), n_bases + (add_intercept))`
    """

    def __init__(self, start=0, end=1, n_bases=10, spline_order=3):

        self.start = start
        self.end = end
        self.n_bases = n_bases
        self.spline_order = spline_order

        self.knots = get_knots(self.start, self.end, self.n_bases, self.spline_order)

        self.S = get_S(self.n_bases, self.spline_order, add_intercept=False)

    def __repr__(self):
        return "BSpline(start={0}, end={1}, n_bases={2}, spline_order={3})".\
            format(self.start, self.end, self.n_bases, self.spline_order)

    def getS(self, add_intercept=False):
        """Get the penalty matrix S

        Returns
            np.array, of shape (n_bases + add_intercept, n_bases + add_intercept)
        """
        S = self.S
        if add_intercept is True:
            # S <- cbind(0, rbind(0, S)) # in R
            zeros = np.zeros_like(S[:1, :])
            S = np.vstack([zeros, S])

            zeros = np.zeros_like(S[:, :1])
            S = np.hstack([zeros, S])
        return S

    def predict(self, x, add_intercept=False):
        """For some x, predict the bn(x) for each base

        Arguments:
            x: np.array; Vector of dimension 1
            add_intercept: bool; should we add the intercept to the final array

        Returns:
            np.array, of shape (len(x), n_bases + (add_intercept))
        """
        # sanity check
        if x.min() < self.start:
            raise Warning("x.min() < self.start")
        if x.max() > self.end:
            raise Warning("x.max() > self.end")

        return get_X_spline(x=x,
                            knots=self.knots,
                            n_bases=self.n_bases,
                            spline_order=self.spline_order,
                            add_intercept=add_intercept)

    def get_config(self):
        return {"start": self.start,
                "end": self.end,
                "n_bases": self.n_bases,
                "spline_order": self.spline_order
                }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

############################################
# core functions


def get_gam_splines(start=0, end=100, n_bases=10, spline_order=3, add_intercept=True):
    """Main function required by (TF)Concise class
    """
    # make sure n_bases is an int
    assert type(n_bases) == int

    x = np.arange(start, end + 1)

    knots = get_knots(start, end, n_bases, spline_order)
    X_splines = get_X_spline(x, knots, n_bases, spline_order, add_intercept)
    S = get_S(n_bases, spline_order, add_intercept)
    # Get the same knot positions as with mgcv
    # https://github.com/cran/mgcv/blob/master/R/smooth.r#L1560

    return X_splines, S, knots


############################################
# helper functions
# main resource:
# https://github.com/cran/mgcv/blob/master/R/smooth.r#L1560
def get_knots(start, end, n_bases=10, spline_order=3):
    """
    Arguments:
        x; np.array of dim 1
    """
    x_range = end - start
    start = start - x_range * 0.001
    end = end + x_range * 0.001

    # mgcv annotation
    m = spline_order - 1
    nk = n_bases - m            # number of interior knots

    dknots = (end - start) / (nk - 1)
    knots = np.linspace(start=start - dknots * (m + 1),
                        stop=end + dknots * (m + 1),
                        num=nk + 2 * m + 2)
    return knots.astype(np.float32)


# - get knots as arguments
def get_X_spline(x, knots, n_bases=10, spline_order=3, add_intercept=True):
    """
    Returns:
        np.array of shape [len(x), n_bases + (add_intercept)]

    # BSpline formula
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html#scipy.interpolate.BSpline

    Fortran code:
    https://github.com/scipy/scipy/blob/v0.19.0/scipy/interpolate/fitpack/splev.f
    """
    if len(x.shape) is not 1:
        raise ValueError("x has to be 1 dimentional")

    tck = [knots, np.zeros(n_bases), spline_order]

    X = np.zeros([len(x), n_bases])

    for i in range(n_bases):
        vec = np.zeros(n_bases)
        vec[i] = 1.0
        tck[1] = vec

        X[:, i] = si.splev(x, tck, der=0)

    if add_intercept is True:
        ones = np.ones_like(X[:, :1])
        X = np.hstack([ones, X])

    return X.astype(np.float32)


def get_S(n_bases=10, spline_order=3, add_intercept=True):
    # mvcv R-code
    # S<-diag(object$bs.dim);
    # if (m[2]) for (i in 1:m[2]) S <- diff(S)
    # object$S <- list(t(S)%*%S)  # get penalty
    # object$S[[1]] <- (object$S[[1]]+t(object$S[[1]]))/2 # exact symmetry

    S = np.identity(n_bases)
    m2 = spline_order - 1  # m[2] is the same as m[1] by default

    # m2 order differences
    for i in range(m2):
        S = np.diff(S, axis=0)  # same as diff() in R

    S = np.dot(S.T, S)
    S = (S + S.T) / 2  # exact symmetry

    if add_intercept is True:
        # S <- cbind(0, rbind(0, S)) # in R
        zeros = np.zeros_like(S[:1, :])
        S = np.vstack([zeros, S])

        zeros = np.zeros_like(S[:, :1])
        S = np.hstack([zeros, S])

    return S.astype(np.float32)
