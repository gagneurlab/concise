import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.layers.pooling import _GlobalPooling1D
from keras.layers import Conv1D, Input
from concise.regularizers import GAMRegularizer
from concise.splines import BSpline
import matplotlib.pyplot as plt


# TODO - improve the naming
# TODO - unit-tests for the general case of smoothing: encodeSplines
# TODO - write unit-tests - use synthetic dataset from motifp - check both cases encodeSplines

############################################


def InputDNA(seq_length, name=None, **kwargs):
    """Convenience wrapper around keras.layers.Input:

    Input((seq_length, 4), name=name, **kwargs)
    """
    return Input((seq_length, 4), name=name, **kwargs)


def InputDNAQuantity(seq_length, n_features=1, name=None, **kwargs):
    """Convenience wrapper around keras.layers.Input:

    Input((seq_length, n_features), name=name, **kwargs)
    """
    return Input((seq_length, n_features), name=name, **kwargs)


def InputDNAQuantitySplines(seq_length, n_bases, name="DNASmoothPosition", **kwargs):
    """Convenience wrapper around keras.layers.Input:

    Input((seq_length, n_bases), name=name, **kwargs)
    """
    return Input((seq_length, n_bases), name=name, **kwargs)


class GlobalSumPooling1D(_GlobalPooling1D):
    """Global average pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        return K.sum(inputs, axis=1)


class ConvDNA(Conv1D):
    """
    Convenience wrapper over keras.layers.Conv1D with 2 changes:
    - additional argument seq_length specifying input_shape
    - restriction in build method: input_shape[-1] needs to be 4
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seq_length=None,
                 **kwargs):

        # override input shape
        if seq_length:
            kwargs["input_shape"] = (seq_length, 4)
            kwargs["batch_input_shape"] = None

        super(ConvDNA, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.seq_length = seq_length

    def build(self, input_shape):
        if input_shape[-1] is not 4:
            raise ValueError("ConvDNA requires input_shape[-1] == 4")
        return super(ConvDNA, self).build(input_shape)

    def get_config(self):
        config = super(ConvDNA, self).get_config()
        config["seq_length"] = self.seq_length
        return config

    # TODO - define the plotting function for motifs

############################################
# Smoothing layers

# TODO - add X_spline as non-trainable weights
class GAMSmooth(Layer):

    def __name__(self):
        return "GAMSmooth"

    def __init__(self,
                 # spline type
                 n_bases=10,
                 spline_order=3,
                 share_splines=False,
                 spline_exp=False,
                 # regularization
                 l2_smooth=1e-5,
                 l2=1e-5,
                 use_bias=False,
                 bias_initializer='zeros',
                 **kwargs):
        """

        Arguments:
            n_splines int: Number of splines used for the positional bias.
            spline_exp (bool): If True, the positional bias score is observed by: :code:`np.exp(spline_score)`,
                  where :code:`spline_score` is the linear combination of B-spline basis functions. If False, :code:`np.exp(spline_score + 1)` is used.
            l2 (float): L2 regularization strength for the second order differences in positional bias' smooth splines. (GAM smoothing regularization)
            l2_smooth (float): L2 regularization strength for the spline base coefficients.
            use_bias: boolean; should we add a bias to the transition
            bias_initializer; bias initializer - from keras.initailizers
        """
        self.n_bases = n_bases
        self.spline_order = spline_order
        self.share_splines = share_splines
        self.spline_exp = spline_exp
        self.l2 = l2
        self.l2_smooth = l2_smooth
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)

        super(GAMSmooth, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (None, steps, filters)

        start = 0
        end = input_shape[1]
        filters = input_shape[2]

        if self.share_splines:
            n_spline_tracks = 1
        else:
            n_spline_tracks = filters

        # setup the bspline object
        self.bs = BSpline(start, end - 1,
                          n_bases=self.n_bases,
                          spline_order=self.spline_order
                          )

        # create X_spline,
        self.positions = np.arange(end)
        self.X_spline = self.bs.predict(self.positions, add_intercept=False)  # shape = (end, self.n_bases)

        # convert to the right precision and K.constant
        self.X_spline_K = K.constant(K.cast_to_floatx(self.X_spline))

        # add weights
        self.kernel = self.add_weight(shape=(self.n_bases, n_spline_tracks),
                                      initializer='ones',
                                      name='kernel',
                                      regularizer=GAMRegularizer(self.n_bases, self.spline_order,
                                                                 self.l2_smooth, self.l2),
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight((n_spline_tracks, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None)

        super(GAMSmooth, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        spline_track = K.dot(self.X_spline_K, self.kernel)

        if self.use_bias:
            spline_track = K.bias_add(spline_track, self.bias)

        if self.spline_exp:
            spline_track = K.exp(spline_track)
        else:
            spline_track = spline_track + 1

        # multiply together the two coefficients
        output = spline_track * x

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'n_bases': self.n_bases,
            'spline_order': self.spline_order,
            'share_splines': self.share_splines,
            'spline_exp': self.spline_exp,
            'l2_smooth': self.l2_smooth,
            'l2': self.l2,
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(GAMSmooth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def positional_effect(self):
        w = self.get_weights()[0]
        pos_effect = np.dot(self.X_spline, w)
        return {"positional_effect": pos_effect, "positions": self.positions}

    def plot(self, *args, **kwargs):
        pe = self.positional_effect()
        plt.plot(pe["positions"], pe["positional_effect"], *args, **kwargs)
        plt.xlabel("Position")
        plt.ylabel("Positional effect")

        # TODO - define a plotting function - plot f(x)


class ConvDNAQuantitySplines(Conv1D):
    """
    Convenience wrapper over keras.layers.Conv1D with 2 changes:
    - additional argument seq_length specifying input_shape (as in ConvDNA)
    - restriction in kernel_regularizer - needs to be of class GAMRegularizer
    - hard-coded values:
       - kernel_size=1,
       - strides=1,
       - padding='valid',
       - dilation_rate=1,
    """

    def __init__(self, filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=GAMRegularizer(),
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 seq_length=None,
                 **kwargs):

        # override input shape
        if seq_length:
            kwargs["input_shape"] = (seq_length, 4)
            kwargs["batch_input_shape"] = None

        # require GAMRegularizer
        if not isinstance(kernel_regularizer, GAMRegularizer):
            raise ValueError("Regularizer has to be of type concise.regularizers.GAMRegularizer")

        super(ConvDNAQuantitySplines, self).__init__(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='valid',
            dilation_rate=1,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.seq_length = seq_length

    def build(self, input_shape):
        # update the regularizer
        self.kernel_regularizer.n_bases = input_shape[2]

        return super(ConvDNAQuantitySplines, self).build(input_shape)

    def get_config(self):
        config = super(ConvDNAQuantitySplines, self).get_config()
        config.pop('kernel_size')
        config.pop('strides')
        config.pop('padding')
        config.pop('dilation_rate')
        config["seq_length"] = self.seq_length
        return config
