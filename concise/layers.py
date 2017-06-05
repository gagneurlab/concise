import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras.layers.pooling import _GlobalPooling1D
from keras.layers import Conv1D, Input
from keras.layers.core import Dropout
from deeplift.visualization import viz_sequence
import matplotlib.pyplot as plt

from concise.utils.pwm import DEFAULT_BASE_BACKGROUND, pssm_array2pwm_array, _pwm2pwm_info
import concise.regularizers as cr
from concise.regularizers import GAMRegularizer
from concise.utils.splines import BSpline
from concise.utils.helper import get_from_module
from concise.utils.plot import heatmap
from concise.preprocessing.sequence import (DNA, RNA, AMINO_ACIDS,
                                            CODONS, STOP_CODONS)
from concise.preprocessing.structure import RNAplfold_PROFILES
# TODO - improve the naming
# TODO - unit-tests for the general case of smoothing: encodeSplines
# TODO - write unit-tests - use synthetic dataset from motifp - check both cases encodeSplines

############################################

# --------------------------------------------
# Input()


def InputDNA(seq_length, name=None, **kwargs):
    """Input placeholder for array returned by `encodeDNA` or `encodeRNA`

    Wrapper for: `keras.layers.Input((seq_length, 4), name=name, **kwargs)`
    """
    return Input((seq_length, 4), name=name, **kwargs)


InputRNA = InputDNA


def InputCodon(seq_length, ignore_stop_codons=True, name=None, **kwargs):
    """Input placeholder for array returned by `encodeCodon`

    Note: The seq_length is divided by 3

    Wrapper for: `keras.layers.Input((seq_length / 3, 61 or 61), name=name, **kwargs)`
    """
    if ignore_stop_codons:
        vocab = CODONS
    else:
        vocab = CODONS + STOP_CODONS

    assert seq_length % 3 == 0
    return Input((seq_length / 3, len(vocab)), name=name, **kwargs)


def InputAA(seq_length, name=None, **kwargs):
    """Input placeholder for array returned by `encodeAA`

    Wrapper for: `keras.layers.Input((seq_length, 22), name=name, **kwargs)`
    """
    return Input((seq_length, len(AMINO_ACIDS)), name=name, **kwargs)


def InputRNAStructure(seq_length, name=None, **kwargs):
    """Input placeholder for array returned by `encodeRNAStructure`

    Wrapper for: `keras.layers.Input((seq_length, 5), name=name, **kwargs)`
    """
    return Input((seq_length, len(RNAplfold_PROFILES)), name=name, **kwargs)


def InputSplines(seq_length, n_bases=10, name=None, **kwargs):
    """Input placeholder for array returned by `encodeSplines`

    Wrapper for: `keras.layers.Input((seq_length, n_bases), name=name, **kwargs)`
    """
    return Input((seq_length, n_bases), name=name, **kwargs)


# TODO - deprecate
def InputDNAQuantity(seq_length, n_features=1, name=None, **kwargs):
    """Convenience wrapper around `keras.layers.Input`:

    `Input((seq_length, n_features), name=name, **kwargs)`
    """
    return Input((seq_length, n_features), name=name, **kwargs)


# TODO - deprecate
def InputDNAQuantitySplines(seq_length, n_bases=10, name="DNASmoothPosition", **kwargs):
    """Convenience wrapper around keras.layers.Input:

    `Input((seq_length, n_bases), name=name, **kwargs)`
    """
    return Input((seq_length, n_bases), name=name, **kwargs)

# --------------------------------------------

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


# TODO how to write a generic class for it?

class ConvSequence(Conv1D):
    """Convenience wrapper over keras.layers.Conv1D with 3 changes:

    - plotting method: plot_weights
    - additional argument seq_length instead of input_shape
    - restriction in build method: input_shape[-1] needs to be the
    same as the vocabulary size
    """

    VOCAB = DNA

    def __init__(self,
                 filters,
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
            kwargs["input_shape"] = (seq_length, len(self.VOCAB))
            kwargs.pop("batch_input_shape", None)

        super(ConvSequence, self).__init__(
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
        if input_shape[-1] is not len(self.VOCAB):
            raise ValueError("{cls} requires input_shape[-1] == {n}. Given: {s}".
                             format(cls=self.__class__.__name__, n=len(self.VOCAB), s=input_shape[-1]))
        return super(ConvSequence, self).build(input_shape)

    def get_config(self):
        config = super(ConvSequence, self).get_config()
        config["seq_length"] = self.seq_length
        return config

    def _plot_weights_heatmap(self, index=None, figsize=(6, 2), **kwargs):
        """Plot weights as a heatmap

        index = can be a particular index or a list of indicies
        **kwargs - additional arguments to concise.utils.plot.heatmap
        """
        W = self.get_weights()[0]
        if index is None:
            index = np.arange(W.shape[2])

        fig = heatmap(np.swapaxes(W[:, :, index], 0, 1), plot_name="filter index: ",
                      vocab=self.VOCAB, figsize=figsize, **kwargs)
        plt.show()
        return fig

    def plot_weights(self, index=None, plot_type="heatmap", figsize=(6, 2), **kwargs):
        """Plot weights as a heatmap

        index = can be a particular index or a list of indicies
        **kwargs - additional arguments to concise.utils.plot.heatmap
        """

        if plot_type == "heatmap":
            return self._plot_weights_heatmap(index=index, figsize=figsize, **kwargs)
        else:
            raise ValueError("plot_type needs to be from {\'heatmap\', \'raw\', \'pwm\', \'pwm_info'}")


class ConvDNA(ConvSequence):
    VOCAB = DNA

    def plot_weights_motif(self, index, plot_type="motif_raw",
                           background_probs=DEFAULT_BASE_BACKGROUND,
                           figsize=(10, 2)):
        """Index can only be a single int
        """

        w_all = self.get_weights()
        if len(w_all) == 0:
            raise Exception("Layer needs to be initialized first")
        W = w_all[0]
        if index is None:
            index = np.arange(W.shape[2])

        if isinstance(index, int):
            index = [index]

        for idx in index:
            w = W[:, :, idx]
            if plot_type == "motif_pwm":
                arr = pssm_array2pwm_array(w[:, :, np.newaxis], background_probs)
            elif plot_type == "motif_raw":
                arr = w
            elif plot_type == "motif_pwm_info":
                quasi_pwm = pssm_array2pwm_array(w[:, :, np.newaxis], background_probs)
                arr = _pwm2pwm_info(np.squeeze(quasi_pwm, -1))
            else:
                raise ValueError("plot_type needs to be from {\'raw\', \'pwm\', \'pwm_info'}")

            if len(index) > 1:
                print("filter index: {0}".format(idx))
            viz_sequence.plot_weights(arr, figsize=figsize)

    def plot_weights(self, index=None, plot_type="motif_raw", figsize=(6, 2), **kwargs):
        """Plot weights as a heatmap

        index = can be a particular index or a list of indicies
        **kwargs - additional arguments to concise.utils.plot.heatmap
        """

        if plot_type == "heatmap":
            return self._plot_weights_heatmap(index=index, figsize=figsize, **kwargs)
        elif plot_type[:5] == "motif":
            return self.plot_weights_motif(index=index, plot_type=plot_type, figsize=figsize, **kwargs)
        else:
            raise ValueError("plot_type needs to be from {\'heatmap\', \'raw\', \'pwm\', \'pwm_info'}")

    # TODO - improve the plotting functions for motifs - refactor the viz_sequence
    #        - mutliple panels with titles
    #        - save to file if needed


class ConvRNA(ConvDNA):
    # TODO - implement the letter U in for plotting
    VOCAB = RNA


class ConvAA(ConvSequence):
    VOCAB = AMINO_ACIDS


class ConvRNAStructure(ConvSequence):
    VOCAB = RNAplfold_PROFILES


class ConvCodon(ConvSequence):
    VOCAB = CODONS

    def build(self, input_shape):
        if input_shape[-1] not in [len(CODONS), len(CODONS + STOP_CODONS)]:
            raise ValueError("{cls} requires input_shape[-1] == {n} or {m}".
                             format(cls=self.__class__.__name__,
                                    n=len(CODONS),
                                    m=len(CODONS + STOP_CODONS)))

        if input_shape[-1] == len(CODONS + STOP_CODONS):
            self.VOCAB = CODONS + STOP_CODONS

        return super(ConvSequence, self).build(input_shape)

# --------------------------------------------

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

        # Arguments:
            n_splines int: Number of splines used for the positional bias.
            spline_exp (bool): If True, the positional bias score is observed by: `np.exp(spline_score)`,
               where `spline_score` is the linear combination of B-spline basis functions.
               If False, `np.exp(spline_score + 1)` is used.
            l2 (float): L2 regularization strength for the second order differences in positional bias' smooth splines.
            (GAM smoothing regularization)
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

        # add weights - all set to 0
        self.kernel = self.add_weight(shape=(self.n_bases, n_spline_tracks),
                                      initializer='zeros',
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


SmoothPositionWeight = GAMSmooth
# SplineWeight ?

# TODO - add the plotting functionality
# TODO - rename the layer
# additional arguments?
# - share_splines=False,
# - spline_exp=False
#
# TODO - use similar arguments to GAMSmooth (not as a thin wrapper around Conv1d)
# TODO - fix & unit-test this layer
class ConvSplines(Conv1D):
    """Convenience wrapper over `keras.layers.Conv1D` with 2 changes:
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
                 **kwargs):

        super(ConvSplines, self).__init__(
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

        if not isinstance(self.kernel_regularizer, cr.GAMRegularizer):
            raise ValueError("Regularizer has to be of type concise.regularizers.GAMRegularizer. " +
                             "Current type: " + str(type(self.kernel_regularizer)),
                             "\nObject: " + str(self.kernel_regularizer))

        # self.seq_length = seq_length

    def build(self, input_shape):
        # update the regularizer
        self.kernel_regularizer.n_bases = input_shape[2]

        return super(ConvSplines, self).build(input_shape)

    def get_config(self):
        config = super(ConvSplines, self).get_config()
        config.pop('kernel_size')
        config.pop('strides')
        config.pop('padding')
        config.pop('dilation_rate')
        # config["seq_length"] = self.seq_length
        return config

class BiDropout(Dropout):
    """Applies Dropout to the input, no matter if in learning phase or not.
    """

    def __init__(self, bi_dropout=True, **kwargs):
        # __init__(self, rate, noise_shape=None, seed=None, **kwargs)
        super(BiDropout, self).__init__(**kwargs)
        self.bi_dropout = bi_dropout

    #
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            #
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            if self.bi_dropout:
                # K.in_train_phase returns the first argument if in training phase otherwise the second
                # return K.in_train_phase(dropped_inputs, inputs, training=training)
                # Taken from keras.backend.tensorflow_backend
                if callable(dropped_inputs):
                    return dropped_inputs()
                else:
                    return dropped_inputs
            else:
                return K.in_train_phase(dropped_inputs, inputs,
                                        training=training)
        return inputs

    #
    @classmethod
    def create_from_dropout(cls, dropout_obj):
        if not isinstance(dropout_obj, Dropout):
            raise Exception("Only Dropout objects can be converted this way!")
        kwargs = dropout_obj.get_config()
        # alternatively can we use "get_config" in combination with (Layer.__init__)allowed_kwargs?
        return cls(**kwargs)

# backcompatibility
ConvDNAQuantitySplines = ConvSplines


AVAILABLE = ["InputDNA", "ConvDNA",
             "InputRNA", "ConvRNA",
             "InputCodon", "ConvCodon",
             "InputAA", "ConvAA",
             "InputRNAStructure", "ConvRNAStructure",
             "InputSplines", "ConvSplines",
             "GlobalSumPooling1D",
             "SmoothPositionWeight",
             # legacy
             "InputDNAQuantitySplines", "InputDNAQuantity",
             "GAMSmooth", "ConvDNAQuantitySplines", "BiDropout"]


def get(name):
    return get_from_module(name, globals())
