"""Template for models
"""

from keras.models import Model
from keras.optimizers import Adam
import keras.layers as kl
import keras.initializers as ki
import keras.regularizers as kr

# concise modules
from concise import initializers as ci
from concise import layers as cl
from concise.utils import PWM


# ### 'First' Concise architecture from Tensorflow

# Splines:
# - `spline_score = X_spline %*% spline_weights`
# - Transform:
#   - `exp(spline_score)`
#   - `spline_score + 1`

# Linear features:
# - `lm_feat = X_feat %*% feature_weights`

# Model:
# - conv2d, `padding = "valid", w = motif_base_weights`
# - activation: exp or relu, bias = motif_bias
# - elementwise_multiply: `hidden * spline_score`
# - pooling: max, sum or mean (accross the whole model)
# - Optionally: multiply by non-linear scaling factor (model fitting)
# - `pool_layer %*% motif_weights + X_feat %*% feature_weights + final_bias`
# - loss: mse
# - optimizer: Adam, optionally l-BFGS

# Regularization:
# - motif_base_weights, L1: motif_lamb
# - motif_weights, L1: lambd
# - spline_weights:
#   - `diag(t(spline_weights) %*% S %*% spline_weights)`, L2_mean: spline_lamb
#   - spline_weights, L2 / n_spline_tracks: spline_param_lamb
# convolution model

def single_layer_pos_effect(pooling_layer="sum",  # 'sum', 'max' or 'mean'
                            nonlinearity="relu",  # 'relu' or 'exp'
                            motif_length=9,
                            n_motifs=6,           # number of filters
                            step_size=0.01,
                            num_tasks=1,          # multi-task learning - 'trans'
                            n_covariates=0,
                            seq_length=100,       # pre-defined sequence length
                            # splines
                            n_splines=None,
                            share_splines=False,  # should the positional bias be shared across motifs
                            # regularization
                            lamb=1e-5,            # overall motif coefficient regularization
                            motif_lamb=1e-5,
                            spline_lamb=1e-5,
                            spline_param_lamb=1e-5,
                            # initialization
                            init_motifs=None,     # motifs to intialize
                            init_motif_bias=0,
                            init_sd_motif=1e-2,
                            init_sd_w=1e-3,       # initial weight scale of feature w or motif w
                            **kwargs):            # unused params

    # initialize conv kernels to known motif pwm's
    if init_motifs:
        # WARNING - initialization is not the same as for Concise class
        pwm_list = [PWM.from_consensus(motif) for motif in init_motifs]
        kernel_initializer = ci.PWMKernelInitializer(pwm_list, stddev=init_sd_motif)
        bias_initializer = ci.PWMBiasInitializer(pwm_list, kernel_size=motif_length)
    else:
        # kernel_initializer = "glorot_uniform"
        kernel_initializer = ki.RandomNormal(stddev=init_sd_motif)
        bias_initializer = ki.Constant(value=init_motif_bias)

    activation = nonlinearity  # supports 'relu' out-of-the-box

    # define the model
    # ----------------
    inputs = []
    seq_input = kl.Input((seq_length, 4))
    inputs.append(seq_input)
    # convolution
    xseq = kl.Conv1D(filters=n_motifs, kernel_size=motif_length,
                     kernel_regularizer=kr.l1(l=motif_lamb),  # Regularization
                     activation=activation,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer
                     )(seq_input)
    # optional positional effect
    if n_splines:
        xseq = cl.GAMSmooth(n_bases=n_splines,
                            share_splines=share_splines,
                            l2_smooth=spline_lamb,
                            l2=spline_param_lamb,
                            )(xseq)
    # pooling layer
    if pooling_layer is "max":
        xseq = kl.pooling.GlobalMaxPooling1D()(xseq)
    elif pooling_layer is "mean":
        xseq = kl.pooling.GlobalAveragePooling1D()(xseq)
    elif pooling_layer is "sum":
        xseq = cl.GlobalSumPooling1D()(xseq)
    else:
        raise ValueError("pooling_layer can only be 'sum', 'mean' or 'max'.")
    # -----
    # add covariates
    if n_covariates:
        cov_input = kl.Input((n_covariates, ))
        inputs.append(cov_input)
        x = kl.concatenate([xseq, cov_input])
    else:
        x = xseq
    # -----

    predictions = kl.Dense(units=num_tasks,
                           kernel_regularizer=kr.l1(lamb),
                           kernel_initializer=ki.RandomNormal(stddev=init_sd_w)
                           )(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=Adam(lr=step_size), loss="mse", metrics=["mse"])

    return model
