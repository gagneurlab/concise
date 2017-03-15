from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import Regularizer

# - TODO what are the kernel shapes?
#    - TODO update pwm_list2array
# - what is x in GAMRegularizer?
#


# TODO - what is x?
# TODO - define better names
class GAMRegularizer(Regularizer):
    """Regularizer for GAM's
    # Arguments
        S: Float matrix; Smoothness matrix - second-order differences
        lamb: Float; Smoothness penalty
        param_lambd: Float; L2 regularization factor. - overall weights regularizer
    """

    def __init__(self, S, lambd=0., param_lambd=0.):
        self.S = K.cast_to_floatx(S)
        self.lambd = K.cast_to_floatx(lambd)
        self.param_lambd = K.cast_to_floatx(param_lambd)

        # TODO - what dimention does X have?

    def __call__(self, x):
        regularization = 0.

        if self.param_lambd:
            regularization += K.sum(self.param_lambd * K.square(x)) / self.S.shape[0]
            # TODO - check if self.S.shape[0] == n_spline_tracks ?!?!

        if self.lambd:
            regularization += self.lambd * K.dot(K.dot(K.transpose(x), self.S), x)
            # https://keras.io/backend/#batch_dot ??
            # TODO - take the diagonal elements

        return regularization

    def get_config(self):
        return {'S': self.S,  # TODO - check the serialization (maybe keras unit-tests?)
                'lambd': float(self.lambd),
                'param_lambd': float(self.param_lambd)}


class GAMSmooth(Layer):
    def __name__(self):
        return "GAMSmooth"

    # TODO - is output_dim the same as filters?
    # TODO - create x_min and x_max
    def __init__(self, output_dim, filters, n_splines,
                 lamb=1e-5, param_lamb=1e-5,
                 share_splines=False,
                 **kwargs):
        self.output_dim = output_dim
        self.filters = filters
        self.n_splines = n_splines
        self.lamb = lamb
        self.param_lamb = param_lamb
        self.share_splines = share_splines

        super(GAMSmooth, self).__init__(**kwargs)

    # TODO - update here
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                 initializer='ones',
                                 regularizer=GAMRegularizer(self.lamb, self.param_lamb),
                                 trainable=True)
        # TODO - X_spline
        X_spline = 1
        # TODO - add the regularizer

        super(GAMSmooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.W)

    # TODO - fix
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'filters': self.filters,
            'n_splines': self.n_splines,
            'lamb': self.lambd,
            'param_lamb': self.param_lamb,
            'share_splines': self.share_splines
        }
        base_config = super(GAMSmooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

