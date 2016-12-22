# get the splines
import numpy as np
import scipy.interpolate as si

def get_gam_splines(start=0, end=100, n_bases=10, spline_order=2, add_intercept=True):
    # make sure n_bases is an int
    assert type(n_bases) == int

    knots = get_knots(start, end, n_bases, spline_order)
    X_splines = get_X_spline(start, end, n_bases, spline_order, add_intercept)
    S = get_S(start, end, n_bases, spline_order, add_intercept)
    # Get the same knot positions as with mgcv
    # https://github.com/cran/mgcv/blob/master/R/smooth.r#L1560

    return X_splines, S, knots


############################################
# helper functions
# main resource:
# https://github.com/cran/mgcv/blob/master/R/smooth.r#L1560
def get_knots(start=0, end=100, n_bases=10, spline_order=2):
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


# TODO - use x = np.linspace(start, end, end - start + 1) as argument instead of start, end
def get_X_spline(start=0, end=100, n_bases=10, spline_order=2, add_intercept=True):
    knots = get_knots(start, end, n_bases, spline_order)
    tck = [knots, np.zeros(n_bases), spline_order]
    x = np.linspace(start, end, end - start + 1)

    X = np.zeros([end - start + 1, n_bases])

    for i in range(n_bases):
        vec = np.zeros(n_bases)
        vec[i] = 1.0
        tck[1] = vec

        X[:, i] = si.splev(x, tck, der=0)

    if add_intercept is True:
        ones = np.ones_like(X[:, :1])
        X = np.hstack([ones, X])

# if (add_intercept == TRUE) {
#     X <- cbind(1, X)
#     ## don't penalize the intercept term
#     S <- cbind(0, rbind(0, S))
#   }
    return X.astype(np.float32)

def get_S(start=0, end=100, n_bases=10, spline_order=2, add_intercept=True):
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
