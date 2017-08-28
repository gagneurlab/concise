from keras import backend as K
import keras.regularizers as kr
from keras.regularizers import Regularizer, serialize, deserialize
from concise.utils.splines import get_S
from concise.utils.helper import get_from_module


class SplineSmoother(Regularizer):
    """Smoothness regularizer for spline transformation.

    It penalizes the differences of neighbouring coefficients.

    # Arguments
        diff_order: neighbouring coefficient difference order
           (2 for second-order differences)
        l2_smooth: float; Non-smoothness penalty (penalize w' * S * w)
        l2: float; L2 regularization factor - overall weights regularizer
    """

    def __init__(self, diff_order=2, l2_smooth=0., l2=0.):
        # convert S to numpy-array if it's a list

        self.diff_order = diff_order
        self.l2_smooth = K.cast_to_floatx(l2_smooth)
        self.l2 = K.cast_to_floatx(l2)

        # convert to K.constant
        self.S = None

    def __call__(self, x):
        # import pdb
        # pdb.set_trace()

        # x.shape = (n_bases, n_spline_tracks)
        # from conv: (kernel_width=1, n_bases, n_spline_tracks)
        from_conv = len(K.int_shape(x)) == 3
        if self.S is None:
            self.S = K.constant(
                K.cast_to_floatx(
                    get_S(K.int_shape(x)[-2], self.diff_order + 1, add_intercept=False)
                ))

        if from_conv:
            x = K.squeeze(x, 0)

        n_spline_tracks = K.cast_to_floatx(K.int_shape(x)[1])

        regularization = 0.

        if self.l2:
            regularization += K.sum(self.l2 * K.square(x)) / n_spline_tracks

        if self.l2_smooth:
            # https://keras.io/backend/#batch_dot
            # equivalent to mean( diag(x' * S * x) )
            regularization += self.l2_smooth * K.mean(K.batch_dot(x, K.dot(self.S, x), axes=1))

        return regularization

    def get_config(self):
        # convert S to list()
        return {'diff_order': self.diff_order,
                'l2_smooth': float(self.l2_smooth),
                'l2': float(self.l2),
                }


# OLD - to be deprecated
class GAMRegularizer(Regularizer):

    def __init__(self, n_bases=10, spline_order=3, l2_smooth=0., l2=0.):
        """Regularizer for GAM's

        # Arguments
            n_bases: number of b-spline bases
            order: spline order (2 for quadratic, 3 for qubic splines)
            l2_smooth: float; Smoothness penalty (penalize w' * S * w)
            l2: float; L2 regularization factor - overall weights regularizer
        """
        # convert S to numpy-array if it's a list

        self.n_bases = n_bases
        self.spline_order = spline_order
        self.l2_smooth = K.cast_to_floatx(l2_smooth)
        self.l2 = K.cast_to_floatx(l2)

        # convert to K.constant
        self.S = K.constant(
            K.cast_to_floatx(
                get_S(n_bases, spline_order, add_intercept=False)
            ))

    def __call__(self, x):
        # x.shape = (n_bases, n_spline_tracks)
        # from conv: (kernel_width=1, n_bases, n_spline_tracks)
        from_conv = len(K.int_shape(x)) == 3
        if from_conv:
            x = K.squeeze(x, 0)

        n_spline_tracks = K.cast_to_floatx(K.int_shape(x)[1])

        regularization = 0.

        if self.l2:
            regularization += K.sum(self.l2 * K.square(x)) / n_spline_tracks

        if self.l2_smooth:
            # https://keras.io/backend/#batch_dot
            # equivalent to mean( diag(x' * S * x) )
            regularization += self.l2_smooth * K.mean(K.batch_dot(x, K.dot(self.S, x), axes=1))

        return regularization

    def get_config(self):
        # convert S to list()
        return {'n_bases': self.n_bases,
                'spline_order': self.spline_order,
                'l2_smooth': float(self.l2_smooth),
                'l2': float(self.l2),
                }


AVAILABLE = ["GAMRegularizer", "SplineSmoother"]


def get(name):
    try:
        return kr.get(name)
    except ValueError:
        return get_from_module(name, globals())
