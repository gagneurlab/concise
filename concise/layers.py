import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.pooling import _GlobalPooling1D
from keras.layers import Conv1D, Input, LocallyConnected1D
from keras.layers.core import Dropout
from concise.utils.plot import seqlogo, seqlogo_fig
import matplotlib.pyplot as plt
from keras.engine import InputSpec

from concise.utils.pwm import DEFAULT_BASE_BACKGROUND, pssm_array2pwm_array, _pwm2pwm_info
from keras import activations
from keras import constraints
from concise import initializers
from concise import regularizers
from concise.regularizers import GAMRegularizer, SplineSmoother
from concise.utils.splines import BSpline
from concise.utils.helper import get_from_module
from concise.utils.plot import heatmap
from concise.preprocessing.sequence import (DNA, RNA, AMINO_ACIDS,
                                            CODONS, STOP_CODONS)
from concise.preprocessing.structure import RNAplfold_PROFILES


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


# deprecated
def InputSplines(seq_length, n_bases=10, name=None, **kwargs):
    """Input placeholder for array returned by `encodeSplines`

    Wrapper for: `keras.layers.Input((seq_length, n_bases), name=name, **kwargs)`
    """
    return Input((seq_length, n_bases), name=name, **kwargs)


def InputSplines1D(seq_length, n_bases=10, name=None, **kwargs):
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

    # Note
      - Input shape: 3D tensor with shape: `(batch_size, steps, features)`.
      - Output shape: 2D tensor with shape: `(batch_size, channels)`

    """

    def call(self, inputs):
        return K.sum(inputs, axis=1)


class ConvSequence(Conv1D):
    """Convenience wrapper over `keras.layers.Conv1D` with 3 changes:

    - additional plotting method: `plot_weights(index=None, plot_type="motif_raw", figsize=None, ncol=1)`
            - **index**: can be a particular index or a list of indicies
            - **plot_type**: Can be one of `"heatmap"`, `"motif_raw"`, `"motif_pwm"` or `"motif_pwm_info"`.
            - **figsize**: tuple, Figure size
            - **ncol**: Number of axis columns
    - additional argument `seq_length` instead of `input_shape`
    - restriction in build method: `input_shape[-1]` needs to the match the vocabulary size

    Clasess `Conv*` all inherit from `ConvSequence` and define the corresponding vocabulary:

    - ConvDNA
    - ConvRNA
    - ConvRNAStructure
    - ConvAA
    - ConvCodon
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

    def _plot_weights_heatmap(self, index=None, figsize=None, **kwargs):
        """Plot weights as a heatmap

        index = can be a particular index or a list of indicies
        **kwargs - additional arguments to concise.utils.plot.heatmap
        """
        W = self.get_weights()[0]
        if index is None:
            index = np.arange(W.shape[2])

        fig = heatmap(np.swapaxes(W[:, :, index], 0, 1), plot_name="filter: ",
                      vocab=self.VOCAB, figsize=figsize, **kwargs)
        # plt.show()
        return fig

    def _plot_weights_motif(self, index, plot_type="motif_raw",
                            background_probs=DEFAULT_BASE_BACKGROUND,
                            ncol=1,
                            figsize=None):
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
        fig = plt.figure(figsize=figsize)

        if plot_type == "motif_pwm" and plot_type in self.AVAILABLE_PLOTS:
            arr = pssm_array2pwm_array(W, background_probs)
        elif plot_type == "motif_raw" and plot_type in self.AVAILABLE_PLOTS:
            arr = W
        elif plot_type == "motif_pwm_info" and plot_type in self.AVAILABLE_PLOTS:
            quasi_pwm = pssm_array2pwm_array(W, background_probs)
            arr = _pwm2pwm_info(quasi_pwm)
        else:
            raise ValueError("plot_type needs to be from {0}".format(self.AVAILABLE_PLOTS))

        fig = seqlogo_fig(arr, vocab=self.VOCAB_name, figsize=figsize, ncol=ncol, plot_name="filter: ")
        # fig.show()
        return fig

    def plot_weights(self, index=None, plot_type="motif_raw", figsize=None, ncol=1, **kwargs):
        """Plot filters as heatmap or motifs

        index = can be a particular index or a list of indicies
        **kwargs - additional arguments to concise.utils.plot.heatmap
        """

        if "heatmap" in self.AVAILABLE_PLOTS and plot_type == "heatmap":
            return self._plot_weights_heatmap(index=index, figsize=figsize, ncol=ncol, **kwargs)
        elif plot_type[:5] == "motif":
            return self._plot_weights_motif(index=index, plot_type=plot_type, figsize=figsize, ncol=ncol, **kwargs)
        else:
            raise ValueError("plot_type needs to be from {0}".format(self.AVAILABLE_PLOTS))


class ConvDNA(ConvSequence):
    VOCAB = DNA
    VOCAB_name = "DNA"
    AVAILABLE_PLOTS = ["heatmap", "motif_raw", "motif_pwm", "motif_pwm_info"]


class ConvRNA(ConvDNA):
    VOCAB = RNA
    VOCAB_name = "RNA"
    AVAILABLE_PLOTS = ["heatmap", "motif_raw", "motif_pwm", "motif_pwm_info"]


class ConvAA(ConvDNA):
    VOCAB = AMINO_ACIDS
    VOCAB_name = "AA"
    AVAILABLE_PLOTS = ["heatmap", "motif_raw"]


class ConvRNAStructure(ConvDNA):
    VOCAB = RNAplfold_PROFILES
    VOCAB_name = "RNAStruct"
    AVAILABLE_PLOTS = ["heatmap", "motif_raw"]


class ConvCodon(ConvSequence):
    VOCAB = CODONS
    AVAILABLE_PLOTS = ["heatmap"]

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


# TODO - re-write SplineWeight1D with SplineT layer
# TODO - SplineWeight1D - use new API and update
#        - think how to call share_splines...?
#        - use a regularizer rather than just
class SplineWeight1D(Layer):
    """Up- or down-weight positions in the activation array of 1D convolutions:

    `x^{out}_{ijk} = x^{in}_{ijk}* (1 + f_S^k(j)) \;,`
    where f_S is the spline transformation.

    # Arguments
        n_bases: int; Number of spline bases used for the positional effect.
        l2_smooth: (float) L2 regularization strength for the second
    order differences in positional bias' smooth splines. (GAM smoothing regularization)
        l2: (float) L2 regularization strength for the spline base coefficients.
        use_bias: boolean; should we add a bias to the transition
        bias_initializer: bias initializer - from `keras.initializers`
    """

    def __name__(self):
        return "SplineWeight1D"

    def __init__(self,
                 # spline type
                 n_bases=10,
                 spline_degree=3,
                 share_splines=False,
                 # regularization
                 l2_smooth=0,
                 l2=0,
                 use_bias=False,
                 bias_initializer='zeros',
                 **kwargs):
        self.n_bases = n_bases
        self.spline_degree = spline_degree
        self.share_splines = share_splines
        self.l2 = l2
        self.l2_smooth = l2_smooth
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)

        super(SplineWeight1D, self).__init__(**kwargs)

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
                          spline_order=self.spline_degree
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
                                      regularizer=GAMRegularizer(self.n_bases, self.spline_degree,
                                                                 self.l2_smooth, self.l2),
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight((n_spline_tracks, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None)

        super(SplineWeight1D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        spline_track = K.dot(self.X_spline_K, self.kernel)

        if self.use_bias:
            spline_track = K.bias_add(spline_track, self.bias)

        # if self.spline_exp:
        #     spline_track = K.exp(spline_track)
        # else:
        spline_track = spline_track + 1

        # multiply together the two coefficients
        output = spline_track * x

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'n_bases': self.n_bases,
            'spline_degree': self.spline_degree,
            'share_splines': self.share_splines,
            # 'spline_exp': self.spline_exp,
            'l2_smooth': self.l2_smooth,
            'l2': self.l2,
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(SplineWeight1D, self).get_config()
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


# SplineT -> use just locally-connected layers
# SplineT1D -> wrap ConvSplines -> However,
# It's better if we use just matrix multiplications.
# - We can have more control over the inputs
# - I think it's better to explicitly state the dimensions: 1D or so.


class SplineT(Layer):
    """Spline transformation layer.

    As input, it needs an array of scalars pre-processed by `concise.preprocessing.EncodeSplines`
    Specifically, the input/output dimensions are:

    - Input: N x ... x channels x n_bases
    - Output: N x ... x channels

    # Arguments
        shared_weights: bool, if True spline transformation weights
    are shared across different features
        kernel_regularizer: use `concise.regularizers.SplineSmoother`
        other arguments: See `keras.layers.Dense`
    """

    def __init__(self,
                 # regularization
                 shared_weights=False,
                 kernel_regularizer=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs
                 ):
        super(SplineT, self).__init__(**kwargs)  # Be sure to call this somewhere!

        self.shared_weights = shared_weights
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.input_spec = InputSpec(min_ndim=3)

    def build(self, input_shape):
        assert len(input_shape) >= 3

        n_bases = input_shape[-1]
        n_features = input_shape[-2]

        # self.input_shape = input_shape
        self.inp_shape = input_shape
        self.n_features = n_features
        self.n_bases = n_bases

        if self.shared_weights:
            use_n_features = 1
        else:
            use_n_features = self.n_features

        # print("n_bases: {0}".format(n_bases))
        # print("n_features: {0}".format(n_features))

        self.kernel = self.add_weight(shape=(n_bases, use_n_features),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight((n_features, ),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=None)

        self.built = True
        super(SplineT, self).build(input_shape)  # Be sure to call this somewhere!

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def call(self, inputs):
        N = len(self.inp_shape)
        # put -2 axis (features) to the front
        # import pdb
        # pdb.set_trace()

        if self.shared_weights:
            return K.squeeze(K.dot(inputs, self.kernel), -1)

        output = K.permute_dimensions(inputs, (N - 2, ) + tuple(range(N - 2)) + (N - 1,))

        output_reshaped = K.reshape(output, (self.n_features, -1, self.n_bases))
        bd_output = K.batch_dot(output_reshaped, K.transpose(self.kernel))
        output = K.reshape(bd_output, (self.n_features, -1) + self.inp_shape[1:(N - 2)])
        # move axis 0 (features) to back
        output = K.permute_dimensions(output, tuple(range(1, N - 1)) + (0,))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        return output

    def get_config(self):
        config = {
            'shared_weights': self.shared_weights,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        }
        base_config = super(SplineT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # TODO - add X_spline as non-trainable weights


# Deprecated
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

        # Arguments
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


# ResSplineWeight
# SplineWeight ?
# WeightedSum1D

# TODO - add the plotting functionality
# TODO - rename the layer
# additional arguments?
# - share_splines=False,
# - spline_exp=False
#
# TODO - use similar arguments to GAMSmooth (not as a thin wrapper around Conv1d)
# TODO - fix & unit-test this layer

# ConvSplineTr1D
# DenseSplineTr
# SplineTr
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

        if not isinstance(self.kernel_regularizer, regularizers.GAMRegularizer):
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
             "SplineWeight1D",
             "SplineT",
             # legacy
             "InputDNAQuantitySplines", "InputDNAQuantity",
             "GAMSmooth", "ConvDNAQuantitySplines", "BiDropout"]


def get(name):
    return get_from_module(name, globals())
