from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Merge
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l1
from keras.optimizers import Adam

# ### Concise architecture

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

# TODO - define the abstract class

class Method(object):

    # def _args_check(self):

    # def __init__(self, args):
    #         """

    #         """
    #         self.args = args

    # def __str__

    # get_unused_param
    # get_var_initialization
    # _build_graph
    # get_weights
    # get_init_weights
    # _var_res_to_weights
    # get_execution_time
    # get_accuracy
    # is_trained
    # _convert_to_var
    # _get_var_res_sess
    # train
    # _predict_in_session
    # _accuracy_in_session
    # _train_lbfgs
    # _train_adam
    # predict
    # _get_other_var
    # _test
    # print_weights
    # plot_accuracy
    # plot_pos_bias
    # to_dict
    # save
    # _set_var_res
    # from_dict
    # load

    # ConciseCV
    # get_param
    # get_unused_param
    # _get_folds
    # get_folds
    # train
    #


def DNA_conv_layer(seq_length, num_filters=(15, 15), conv_width=(15, 15), pool_width=35, L1=0, dropout=0.1):
    """
    Very frequently used conv layer for sequence
    """
    model = Sequential()
    assert len(num_filters) == len(conv_width)
    for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
        conv_height = 4 if i == 0 else 1
        model.add(Convolution2D(
            nb_filter=nb_filter, nb_row=conv_height,
            nb_col=nb_col, activation='linear',
            init='he_normal', input_shape=(4, seq_length, 1),
            dim_ordering='tf',
            W_regularizer=l1(L1), b_regularizer=l1(L1)))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    # for avg pooling - determine the maximum number of returned dimentions
    # merge together
    # model.add(AveragePooling2D(pool_size=(1, pool_width)))
    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())
    return model


def deep_wide_model(n_features, seq_length, loss="mse", num_filters=(15, 15), conv_width=(15, 15),
                    lr=0.001,
                    pool_width=35, L1=0, L1_weights=0, dropout=0.1):
    conv_model = DNA_conv_layer(seq_length, num_filters, conv_width, pool_width, L1, dropout)

    # linear model
    linear_model = Sequential()
    linear_model.add(Activation("linear", input_shape=(n_features, )))
    merged = Merge([conv_model, linear_model], mode='concat')

    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(output_dim=1, W_regularizer=l1(L1_weights)))

    # model.add(Activation('linear'))
    final_model.compile(optimizer=Adam(lr=lr), loss=loss)
    return final_model


# TODO define new concise layer
