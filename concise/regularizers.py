from keras import backend as K
from keras.regularizers import Regularizer
from concise.splines import get_S


class GAMRegularizer(Regularizer):

    def __init__(self, n_bases=10, spline_order=2, l2_smooth=0., l2=0.):
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
