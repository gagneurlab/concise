from keras import backend as K
from keras.regularizers import Regularizer
import numpy as np
from concise.splines import get_S
# TODO - define better names
# TODO - check correct serialization


class GAMRegularizer(Regularizer):
    """Regularizer for GAM's
    # Arguments
        S: Float matrix; Smoothness matrix - second-order differences
        lamb: Float; Smoothness penalty (penalize w' * S * w)
        param_lambd: Float; L2 regularization factor - overall weights regularizer
    """

    def __init__(self, n_bases=10, spline_order=2, lambd=0., param_lambd=0.):
        # convert S to numpy-array if it's a list

        self.n_bases = n_bases
        self.spline_order = self.spline_order
        self.lambd = K.cast_to_floatx(lambd)
        self.param_lambd = K.cast_to_floatx(param_lambd)

        self.S = K.cast_to_floatx(
            get_S(n_bases, spline_order, add_intercept=False)
        )

    def __call__(self, x):
        # TODO - what dimention does X have?
        regularization = 0.

        print("x.shape = ", x.shape)

        if self.param_lambd:
            regularization += K.sum(self.param_lambd * K.square(x)) / self.S.shape[0]
            # TODO - check if self.S.shape[0] == n_spline_tracks ?!?!

        if self.lambd:
            regularization += self.lambd * K.dot(K.dot(K.transpose(x), self.S), x)
            # https://keras.io/backend/#batch_dot ??
            # TODO - take the diagonal elements

        return regularization

    def get_config(self):
        # convert S to list()
        return {'n_bases': self.n_bases,
                'spline_order': self.spline_order,
                'lambd': float(self.lambd),
                'param_lambd': float(self.param_lambd),
                }
