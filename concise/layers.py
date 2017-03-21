from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from concise.regularizers import GAMRegularizer
import numpy as np
from concise.splines import BSpline


# TODO - implement other functionality as in dense:
# https://github.com/fchollet/keras/blob/master/keras/layers/core.py#L728

# TODO - use pre-processing for GAM's

class GAMSmooth(Layer):
    def __name__(self):
        return "GAMSmooth"

    # TODO - is output_dim the same as filters?
    def __init__(self,
                 x,
                 filters,
                 # spline type
                 n_bases=10,
                 spline_order=3,
                 share_splines=False,
                 # regularization
                 lamb=1e-5,
                 param_lamb=1e-5,
                 use_bias=True,
                 bias_initializer='zeros',
                 **kwargs):
        """

        Arguments
            x: np.array of dimension = 1; all the possible values for the input
        """
        if isinstance(x, list):
            x = np.asarray(x)

        self.x = np.unique(x)
        # self.output_dim = output_dim
        self.filters = filters
        self.n_bases = n_bases
        self.spline_order = spline_order
        self.share_splines = share_splines
        self.lamb = lamb
        self.param_lamb = param_lamb
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)

        # setup the bspline object
        self.bs = BSpline(start=self.x.min(), end=self.x.max(),
                          n_bases=self.n_bases,
                          spline_order=self.spline_order
                          )

        # create X_spline
        self.X_spline = self.bs.predict(self.x, add_intercept=False)

        super(GAMSmooth, self).__init__(**kwargs)

    # TODO - update here - consider multiple channels for the position?
    def build(self, input_shape):

        # TODO - restrict the input only to a 2d input

        # input_shape: (batch_size, seq_length)
        # num_channels = input_shape[-1]  # channels_last

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(self.n_bases, self.filters),
                                      initializer='ones',
                                      name='kernel',
                                      regularizer=GAMRegularizer(self.n_bases, self.spline_order,
                                                                 self.lamb, self.param_lamb),
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight((self.filters, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None)

        # save the input_shape
        self.input_shape = input_shape

        super(GAMSmooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # 1. do I have to take care about the batch dimention?
        # TODO - write the transformation function

        # 1. all x have to exactly match the x's

        # TODO - create the hash function
        x_long = x.reshape((-1))

        np.where(x_long == a2)

        K.unique(x)
        # TODO - there is no unique function in keras...
        x_uniq = np.unique(x)  # 1d version

        # gather(self.X_spline, indices)

        output_shrunken = K.dot(self.X_spline[hash(which_uniq), :], self.kernel)  # (x_uniq, filters)

        # expand the dimensions
        output = output_shrunken[hash(x, key=x_uniq), :]

        # TODO - X_spline long should have the shape: (n_batch, seq_length, n_bases)
        #
        # K.dot(X_spline_x, self.kernel).shape = (n_batch, seq_length, filters)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        return output

    # TODO - check the dimentions
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.filters
        return tuple(output_shape)

    def get_config(self):
        # TODO - save X
        config = {
            'x': self.x.tolist(),
            # 'output_dim': self.output_dim,
            'filters': self.filters,
            'n_bases': self.n_bases,
            'spline_order': self.spline_order,
            'share_splines': self.share_splines,
            'lambd': self.lambd,
            'param_lamb': self.param_lamb,
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(GAMSmooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


