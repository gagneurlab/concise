# -*- coding: utf-8 -*-
# concise end-user interface

# TODO - update process data function & Concise -
#        squeeze 2nd dimension (None, 1, seq_len, 4) -> (None, seq_len, 4)
#

# OLD:
# TODO: write the train method Concise more roboustly, when retraining
# - write additional method continue training
# - train - starts training from scratch?
# TODO: init_motifs input check - don't allow longer motifs than the motif_length
# MAYBE TODO - implement more rich semantics on feature shareing accross conditions
from .legacy import analyze
from .legacy import get_data
from .utils import splines
from .utils import helper
from .utils import tf_helper
from . import eval_metrics as ce
import numpy as np
import tensorflow as tf
import pprint
import json
import os
import inspect
import copy
import time
from sklearn.model_selection import KFold


class Concise(object):
    """**Initialize the Concise object.**

    This is the main class for:

    - fitting the CONCISE model :py:meth:`train`
    - making predictions on a new data-set :py:meth:`predict`
    - saving, loading the model from a file :py:meth:`save`, :py:meth:`load`

    Args:
        pooling_layer (str): Pooling layer to use. Can be :code:`"sum"` or :code:`"max"`.
        nonlinearity (str): Activation function to use after the convolutional layer. Can be :code:`"relu"` or :code:`"exp"`
        optimizer (str): Which optimizer to use. Can be :code:`"adam"` or :code:`"lbfgs"`.
        batch_size (int): Batch size - number of training samples used in one parameter update iteration.
        n_epochs (int): Number of epochs - how many times should a single training sample be used in the parameter update iteration.
        early_stop_patience (int or None): Number of epochs with no improvement after which training will be stopped. If None, don't use early_stop.
        n_iterations_checkpoint (int): Number of internal L-BFGS-B steps to perform at every step.
        motif_length (int): Length of the trained motif (number), i.e. width of the convolutional filter.
        n_motifs (int): Number of motifs to train.
        step_size (float): Step size or learning rate. Size of the parameter update in the ADAM optimizer. Very important tuning parameter.
        step_epoch (int): After how many training epochs should is the :py:attr:`step_size` parameter decreased.
        step_decay (float): To what fraction is the :py:attr:`step_size` reduced after every :py:attr:`step_epoch`, i.e. :code:`step_size *= step_decay`
        num_tasks (int): Number of tasks to perform i.e. number of columns in the response variable y.
        n_splines (int or None): Number of splines used for the positional bias. If :code:`None`, the positional bias is not used.
        share_splines (bool): If True, all the motifs share the same positional bias. If False, each motif has its own positional bias.
        spline_exp (bool): If True, the positional bias score is observed by: :code:`np.exp(spline_score)`, where :code:`spline_score` is the linear combination of B-spline basis functions. If False, :code:`np.exp(spline_score + 1)` is used.
        lamb (float): L1 regularization strength for the additional feature coefficients/weights.
        motif_lamb (float): L1 regularization strength for the motif coefficients/weights.
        spline_lamb (float): L2 regularization strength for the second order differences in positional bias' smooth splines. (GAM smoothing regularization)
        spline_param_lamb (float): L2 regularization strength for the spline base coefficients.
        init_motifs (list of chr): List of motifs used to initialize the model. Their length has to be smaller or equal to :py:attr:`motif_length`. If it is smaller, they will be padded with undefined bases (N's). Number of provided motifs (list length) has to be smaller or equal to :py:attr:`n_motifs`. If it is smaller, :code:`'NN...NN'` will be used for the missing motifs.
        init_motifs_scale (float): Scale at which to initialize the motif weights. If small (close to 1), the provided :py:attr:`init_motifs` will have a small impact.
        nonlinearity_scale_factor (float): Scaling factor after the pooling layer. This is useful in order to have all the weights roughly on the same scale (and hence be able to use a single :py:attr:`step_size`).
        init_motif_bias (float): Initial value for the bias.
        init_sd_motif (float): Standard deviation of the noise added to initialized motif weights.
        init_sd_w (float): Standard deviation of the noise added to initialized feature weights.

        print_every (int): Number of iteration steps after the training information is printed.
        **kwargs (any): Additional unused parameters that get stored in the model file. Useful to store say the pre-processing information.
    """

    # TODO: __repr__
    # TODO: change all N_ uppercase to lowercase for consistency
    # correct types
    _correct_arg_types = {
        "pooling_layer": {str},
        "optimizer": {str},
        "batch_size": {int, np.int64},
        "n_epochs": {int, np.int64},
        "early_stop_patience": {int, type(None)},
        "n_iterations_checkpoint": {int, np.int64},
        "motif_length": {int, np.int64},
        "n_motifs": {int, np.int64},
        "step_size": {float, np.float64, np.float},
        "step_decay": {float, np.float64, np.float},
        "step_epoch": {int, np.int64},
        "n_splines": {int, np.int64, type(None)},
        "share_splines": {bool},
        "lamb": {float, np.float64},
        "motif_lamb": {float, np.float64},
        "spline_lamb": {float, np.float64},
        "spline_param_lamb": {float, np.float64},
        "init_motifs": {str, tuple, list, type(None)},  # motifs to intialize
        "init_motif_bias": {list, np.ndarray, float, int, np.int64, type(None), np.float64},
        "init_sd_motif": {float, np.float64},
        "init_sd_w": {float, np.float64},         # initial weight scale of feature w or motif w
        "print_every": {int, np.int64}
    }

    def _args_check(self):
        for key, value in self._param.items():
            if key in Concise._correct_arg_types.keys() and not type(value) in Concise._correct_arg_types[key]:
                raise TypeError("argument \"" + str(key) +
                                "\" has to be of type " +
                                str(Concise._correct_arg_types[key]))

    def __init__(self,
                 pooling_layer="sum",
                 nonlinearity="relu",  # relu or exp
                 optimizer="adam",
                 batch_size=32,
                 n_epochs=3,
                 early_stop_patience=None,
                 n_iterations_checkpoint=20,
                 # network details
                 motif_length=9,
                 n_motifs=6,
                 step_size=0.01,
                 step_epoch=10,
                 step_decay=0.95,
                 # - multi-task learning
                 num_tasks=1,
                 # - splines
                 n_splines=None,
                 share_splines=False,  # should the positional bias be shared across motifs
                 spline_exp=False,    # use the exponential function
                 # regularization
                 lamb=1e-5,
                 motif_lamb=1e-5,
                 spline_lamb=1e-5,
                 spline_param_lamb=1e-5,
                 # initialization
                 init_motifs=None,  # motifs to intialize
                 init_motifs_scale=1,  # scale at which to initialize the weights
                 # right scale
                 nonlinearity_scale_factor=1,
                 init_motif_bias=0,
                 init_sd_motif=1e-2,
                 init_sd_w=1e-3,         # initial weight scale of feature w or motif w
                 # outuput detail
                 print_every=100,
                 **kwargs):

        self._param = locals()  # get all the init arguments
        del self._param['self']  # remove self from parameters
        del self._param['kwargs']
        self._args_check()
        # self._input_param = self._param
        # general class variables
        self._num_tasks = num_tasks
        self._num_channels = 4
        self._model_fitted = False
        self._exec_time = None
        self._weights = None
        self._accuracy = None

        self.unused_param = kwargs

        # setup splines
        self._splines = None
        if self._param["n_splines"] is not None:
            # if we share the splines, use n_motifs tracks
            if self._param["share_splines"]:
                self._param["n_spline_tracks"] = 1
            else:
                self._param["n_spline_tracks"] = self._param["n_motifs"]

    def __str__(self):

        print_dict = {key: value for key, value in self._param.items() if key not in self.unused_param.keys()}
        print_dict.pop("kwargs", None)

        return pprint.pformat(print_dict)
        # DONE - initialize only at training

    def get_param(self):
        """
        Returns:
            dict: Model's parameter list.
        """
        return self._param

    def get_unused_param(self):
        """
        Returns:
            dict: Model's additional parameters specified with :py:attr:`**kwargs` in :py:meth:`__init__`.
        """
        return self.unused_param

    def _get_var_initialization(self, graph, X_feat_train, y_train):
        # setup filter initial

        # motif -> filter_array
        if self._param["init_motifs"] is not None:
            motifs = get_data.adjust_motifs(self._param["init_motifs"],
                                            self._param["motif_length"],
                                            self._param["n_motifs"])
            filter_init_array = get_data.intial_motif_filter(motifs)
            filter_init_array *= self._param["init_motifs_scale"]  # scale the intial array
            del motifs

        # update filter bias according to the filter_array width
        if self._param["init_motif_bias"] is None and self._param["init_motifs"] is not None:
            # Initialization logic:
            # - 2 missmatches or more should yield a *negative* score,
            # - 1 missmatch or less should yield a *positive* score:
            biases = np.sum(np.amax(filter_init_array, axis=2), axis=1)[0]
            self._param["init_motif_bias"] = - (biases - 1.5 * self._param["init_motifs_scale"])
            del biases
        else:
            self._param["init_motif_bias"] = 0

        feature_weights_init, final_bias_init = (0, 0)

        # Finally, define the variables
        with graph.as_default():
            # Define variables.
            # --------------------------------------------
            # 1. convolution filter
            motif_base_weights = tf.Variable(tf.truncated_normal(
                [1, self._param["motif_length"], self._num_channels, self._param["n_motifs"]],
                mean=0, stddev=self._param["init_sd_motif"]), name="tf_motif_base_weights"
            )

            # intialize around the known motif
            if self._param["init_motifs"] is not None:
                motif_base_weights += filter_init_array

            # + ReLU's
            motif_bias = tf.Variable(tf.zeros([self._param["n_motifs"]]), name="tf_motif_bias") + \
                self._param["init_motif_bias"]

            # --------------------------------------------
            # initalize spline weights
            if self._param["n_splines"] is not None:
                spline_weights = tf.Variable(tf.truncated_normal([self._param["n_splines"],
                                                                  self._param["n_spline_tracks"]],
                                                                 mean=0, stddev=0.1),
                                             name="tf_spline_weights")
                # spline_bias = tf.Variable(tf.ones([1, 1]))  # bias fixed to 1
            else:
                spline_weights = None

            # --------------------------------------------
            # NN
            # motif weights
            motif_weights = tf.Variable(tf.truncated_normal([self._param["n_motifs"], self._num_tasks],
                                                            mean=0.0,
                                                            stddev=self._param["init_sd_w"]),
                                        name="tf_motif_weights"
                                        )

            # feature weights
            # TODO - check here if you are share-ing the weights of features
            feature_weights = tf.Variable(tf.truncated_normal([self._param["n_add_features"], self._num_tasks],
                                                              mean=0,
                                                              stddev=self._param["init_sd_w"],
                                                              dtype=tf.float32),
                                          name="tf_feature_weights")
            feature_weights = feature_weights + feature_weights_init

            final_bias = tf.Variable(tf.constant(final_bias_init,
                                                 shape=[self._num_tasks],
                                                 dtype=tf.float32),
                                     name="tf_final_bias")
            # --------------------------------------------
            # store all the initalized variables to a dictionary
            var = {
                "motif_base_weights": motif_base_weights,
                "motif_bias": motif_bias,
                "spline_weights": spline_weights,
                "feature_weights": feature_weights,
                "motif_weights": motif_weights,
                "final_bias": final_bias
            }

        return var

    def _build_graph(self, graph, var):
        with graph.as_default():
            tf_X_seq = tf.placeholder(tf.float32, shape=[None, 1, None, self._num_channels])
            tf_X_feat = tf.placeholder(tf.float32, shape=[None, None])
            tf_y = tf.placeholder(tf.float32, shape=(None, self._num_tasks))

            # spline initial variable
            if self._param["n_splines"] is not None:
                tf_X_spline = tf.constant(self._splines["X_spline"], dtype=tf.float32)
                S = tf.constant(self._splines["S"], dtype=tf.float32)

            # Model:
            # 1 x (2d conv + Relu)
            # reshape the data into vector
            # 1 x NN

            def model(data, tf_X_feat, var):
                conv = tf.nn.conv2d(data, var["motif_base_weights"],
                                    strides=[1, 1, 1, 1], padding='VALID', name="conv")

                # use the non-linearity
                if self._param["nonlinearity"] == "exp":
                    hidden = tf.exp(conv + var["motif_bias"])
                elif self._param["nonlinearity"] == "relu":
                    hidden = tf.nn.relu(conv + var["motif_bias"])
                else:
                    raise Warning("nonlinearity parameter not valid. Using relu")
                    hidden = tf.nn.relu(conv + var["motif_bias"])

                # multiply by positional bias
                if self._param["n_splines"] is not None:
                    spline_score = tf.matmul(tf_X_spline, var["spline_weights"])
                    # introduce two new axis in the front
                    spline_score = tf.expand_dims(tf.expand_dims(spline_score, 0), 0)

                    if self._param["spline_exp"] == True:
                        hidden = hidden * tf.exp(spline_score)
                    else:
                        hidden = hidden * (spline_score + 1)

                # could do sum or average
                if self._param["pooling_layer"] == "max":
                    pool_layer = tf.reduce_max(hidden, reduction_indices=[1, 2])
                elif self._param["pooling_layer"] == "mean":
                    pool_layer = tf.reduce_mean(hidden, reduction_indices=[1, 2])
                elif self._param["pooling_layer"] == "sum":
                    pool_layer = tf.reduce_sum(hidden, reduction_indices=[1, 2])
                else:
                    raise Warning("unknown pooling_layer use mean")
                    pool_layer = tf.reduce_mean(hidden, reduction_indices=[1, 2])

                # scale the pool_layer with a certain factor for easier initialization / training
                pool_layer *= self._param["nonlinearity_scale_factor"]

                # train a NN on top of it
                return tf.matmul(pool_layer, var["motif_weights"]) + \
                    tf.matmul(tf_X_feat, var["feature_weights"]) + var["final_bias"]

            # define the loss
            y_pred = model(tf_X_seq, tf_X_feat, var)
            # TODO: enable other loss functions like softmax, huber loss etc
            #   - how to include loss-paramteres like k in the case of huber_loss

            # allow for NA's in Y
            # - use tf.is_nan(tf_y) - https://www.tensorflow.org/api_docs/python/tf/is_nan
            # - sum only those values that are none NA
            #    - https://www.tensorflow.org/api_docs/python/tf/where
            y_diff = tf.where(tf.is_nan(tf_y),
                              x=tf.zeros_like(y_pred),
                              y=tf_y - y_pred,
                              name="y_diff")
            loss = tf.reduce_mean(tf.square(y_diff))

            # add regularization
            # regularization = motif_lamb * tf.nn.l2_loss(motif_base_weights) +
            # lamb * tf.nn.l2_loss(motif_weights)
            regularization = self._param["motif_lamb"] * tf_helper.l1_loss(var["motif_base_weights"]) + \
                self._param["lamb"] * tf_helper.l1_loss(var["motif_weights"])
            loss += regularization

            if self._param["n_splines"] is not None:
                # extract the diagonal, take the mean
                spline_penalties = tf.diag_part(tf.matmul(var["spline_weights"],
                                                          tf.matmul(S, var["spline_weights"]),
                                                          transpose_a=True))
                loss += self._param["spline_lamb"] * tf.reduce_mean(spline_penalties)

                # divide by the number of used tracks
                loss += self._param["spline_param_lamb"] * \
                    tf.nn.l2_loss(var["spline_weights"]) / self._param["n_spline_tracks"]
                # reshape to scalar, not required any more...
                # loss = tf.reshape(loss, [])

                # compute the matrix X = dy/dw_spline
                # - you need to feed once per datapoint
                spline_quasi_X = tf.gradients(y_pred, var["spline_weights"])
            else:
                spline_quasi_X = None

            # Optimizer
            # - GradientDescentOptimizer
            # - AdamOptimizer
            tf_step_size = tf.placeholder(tf.float32, shape=[])

            if self._param["optimizer"] == "adam":
                optimizer = tf.train.AdamOptimizer(tf_step_size).minimize(loss)
            elif self._param["optimizer"] == "lbfgs":
                # BFGS loss
                # TODO - implement n_iterations_checkpoint
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    loss, method='L-BFGS-B',
                    options={'maxiter': self._param["n_iterations_checkpoint"]})
            else:
                raise Exception("Optimizer {} not implemented".format(self._param["optimizer"]))
            #
            # http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks

            init = tf.global_variables_initializer()  # tf.initialize_all_variables()

            other_var = {
                "tf_X_feat": tf_X_feat,
                "tf_X_seq": tf_X_seq,
                "tf_y": tf_y,
                "tf_step_size": tf_step_size,
                "y_pred": y_pred,
                "spline_quasi_X": spline_quasi_X,
                "loss": loss,
                "optimizer": optimizer,
                "init": init
            }

        return other_var

    def get_weights(self):
        """
        Returns:
            dict: Model's trained weights.
        """
        if self.is_trained() is False:
            # print("Model not fitted yet. Use object.fit() to fit the model.")
            return None

        var_res = self._var_res
        weights = self._var_res_to_weights(var_res)
        # save to the side
        weights["final_bias_fit"] = weights["final_bias"]
        weights["feature_weights_fit"] = weights["feature_weights"]

        return weights

    def get_init_weights(self):
        """
        Returns:
            dict: Model's initial weights.
        """
        return self.init_weights

    def _var_res_to_weights(self, var_res):
        """
        Get model weights
        """
        # transform the weights into our form
        motif_base_weights_raw = var_res["motif_base_weights"][0]
        motif_base_weights = np.swapaxes(motif_base_weights_raw, 0, 2)

        # get weights
        motif_weights = var_res["motif_weights"]
        motif_bias = var_res["motif_bias"]
        final_bias = var_res["final_bias"]
        feature_weights = var_res["feature_weights"]

        # get the GAM prediction:
        spline_pred = None
        spline_weights = None
        if self._param["n_splines"] is not None:
            spline_pred = self._splines["X_spline"].dot(var_res["spline_weights"])

            if self._param["spline_exp"] is True:
                spline_pred = np.exp(spline_pred)
            else:
                spline_pred = (spline_pred + 1)

            spline_pred.reshape([-1])
            spline_weights = var_res["spline_weights"]

        weights = {"motif_base_weights": motif_base_weights,
                   "motif_weights": motif_weights,
                   "motif_bias": motif_bias,
                   "final_bias": final_bias,
                   "feature_weights": feature_weights,
                   "spline_pred": spline_pred,
                   "spline_weights": spline_weights
                   }
        return weights

    def get_execution_time(self):
        """
        Returns:
            float: Execution time of :py:meth:`train` in seconds.
        """
        return self._exec_time

    def get_accuracy(self):
        """
        Returns:
            dict: Model's accuracies on training.
        """
        if self.is_trained() is False:
            # print("Model not fitted yet. Use object.fit() to fit the model.")
            return None

        return self._accuracy

    # TODO MAYBE -
    # def get_train_history(self):
    #     features = ["loss_history", "step_history", "train_acc_history",
    #                 "val_acc_history", "train_acc_final", "val_acc_final"]
    #     acc = self.get_accuracy()
    #     import pandas as pd
    #     pd.DataFrame(acc[features])

    # def get_test_accuracy(self):
    #     acc = self.get_accuracy()
    #     return acc["test_acc_final"]

    # def get_test_prediction(self):
    #     acc = self.get_accuracy()
    #     return {"test_acc_final": acc["test_acc_final"],
    #             "id_vec_test": acc["id_vec_test"]
    #             }

        # make a DataFrame

    def is_trained(self):
        """
        Returns:
            bool: True if the model already trained, False otherwise.
        """
        return self._model_fitted

    def _get_var_res(self, graph, var, other_var):
        """
        Get the weights from our graph
        """
        with tf.Session(graph=graph) as sess:
            sess.run(other_var["init"])

            # all_vars = tf.all_variables()
            # print("All variable names")
            # print([var.name for var in all_vars])
            # print("All variable values")
            # print(sess.run(all_vars))
            var_res = self._get_var_res_sess(sess, var)
        return var_res

    def _convert_to_var(self, graph, var_res):
        """
        Create tf.Variables from a list of numpy arrays

        var_res: dictionary of numpy arrays with the key names corresponding to var
        """
        with graph.as_default():
            var = {}
            for key, value in var_res.items():
                if value is not None:
                    var[key] = tf.Variable(value, name="tf_%s" % key)
                else:
                    var[key] = None
        return var

    def _get_var_res_sess(self, sess, var):
        var_res = {key: sess.run(value) if value is not None else None for key, value in var.items()}
        return var_res

    # TODO - update this function
    def train(self, X_feat, X_seq, y,
              X_feat_valid=None, X_seq_valid=None, y_valid=None,
              n_cores=3):
        """Train the CONCISE model

        :py:attr:`X_feat`, :py:attr:`X_seq`, py:attr:`y` are preferrably returned by the :py:func:`concise.prepare_data` function.

        Args:
            X_feat: Numpy (float) array of shape :code:`(N, D)`. Feature design matrix storing :code:`N` training samples and :code:`D` features
            X_seq:  Numpy (float) array of shape :code:`(N, 1, N_seq, 4)`. It represents 1-hot encoding of the DNA/RNA sequence.(:code:`N`-seqeuences of length :code:`N_seq`)
            y: Numpy (float) array of shape :code:`(N, 1)`. Response variable.
            X_feat_valid: :py:attr:`X_feat` used for model validation.
            X_seq_valid: :py:attr:`X_seq` used for model validation.
            y: :py:attr:`y` used for model validation.
            n_cores (int): Number of CPU cores used for training. If available, GPU is used for training and this argument is ignored.
        """

        if X_feat_valid is None and X_seq_valid is None and y_valid is None:
            X_feat_valid = X_feat
            X_seq_valid = X_seq
            y_valid = y
            print("Using training samples also for validation ")

        # insert one dimension - backcompatiblity
        X_seq = np.expand_dims(X_seq, axis=1)
        X_seq_valid = np.expand_dims(X_seq_valid, axis=1)

        # TODO: implement the re-training feature
        if self.is_trained() is True:
            print("Model already fitted. Re-training feature not implemented yet")
            return

        # input check
        assert X_seq.shape[0] == X_feat.shape[0] == y.shape[0]
        assert y.shape == (X_feat.shape[0], self._num_tasks)

        # extract data specific parameters
        self._param["seq_length"] = X_seq.shape[2]
        self._param["n_add_features"] = X_feat.shape[1]

        # more input check
        if not self._param["seq_length"] == X_seq_valid.shape[2]:
            raise Exception("sequence lengths don't match")

        # setup splines
        if self._param["n_splines"] is not None:
            padd_loss = self._param["motif_length"] - 1  # how much shorter is our sequence, since we don't use padding
            X_spline, S, _ = splines.get_gam_splines(start=0,
                                                     end=self._param["seq_length"] - padd_loss - 1,  # -1 due to zero-indexing
                                                     n_bases=self._param["n_splines"],
                                                     spline_order=3,
                                                     add_intercept=False)
            self._splines = {"X_spline": X_spline,
                             "S": S
                             }

        # setup graph and variables
        self._graph = tf.Graph()
        self._var = self._get_var_initialization(self._graph, X_feat_train=X_feat, y_train=y)
        self._other_var = self._build_graph(self._graph, self._var)
        # TODO: save the intialized parameters
        var_res_init = self._get_var_res(self._graph, self._var, self._other_var)
        self.init_weights = self._var_res_to_weights(var_res=var_res_init)

        # finally train the model
        # - it saves the accuracy

        if self._param["optimizer"] == "adam":
            _train = self._train_adam
        elif self._param["optimizer"] == "lbfgs":
            _train = self._train_lbfgs
        else:
            raise Exception("Optimizer {} not implemented".format(self._param["optimizer"]))

        self._var_res = _train(X_feat, X_seq, y,
                               X_feat_valid, X_seq_valid, y_valid,
                               graph=self._graph, var=self._var, other_var=self._other_var,
                               early_stop_patience=self._param["early_stop_patience"],
                               n_cores=n_cores)

        self._model_fitted = True

        # TODO: maybe:
        # - add y_train_accuracy
        # - y_train

        return True

    def _predict_in_session(self, sess, other_var, X_feat, X_seq, variable="y_pred"):
        """
        Predict y (or any other variable) from inside the tf session. Variable has to be in other_var
        """
        # other_var["tf_X_seq"]: X_seq, tf_y: y,
        feed_dict = {other_var["tf_X_feat"]: X_feat,
                     other_var["tf_X_seq"]: X_seq}

        y_pred = sess.run(other_var[variable], feed_dict=feed_dict)
        return y_pred

    def _accuracy_in_session(self, sess, other_var, X_feat, X_seq, y):
        """
        Compute the accuracy from inside the tf session
        """
        y_pred = self._predict_in_session(sess, other_var, X_feat, X_seq)
        return ce.mse(y_pred, y)

    def _train_lbfgs(self, X_feat_train, X_seq_train, y_train,
                     X_feat_valid, X_seq_valid, y_valid,
                     graph, var, other_var,
                     early_stop_patience=None,
                     n_cores=3):
        """
        Train the model actual model

        Updates weights / variables, computes and returns the training and validation accuracy
        """
        tic = time.time()
        # take out the parameters for conveience
        n_epochs = self._param["n_epochs"]
        print_every = self._param["print_every"]
        step_size = self._param["step_size"]
        num_steps = n_epochs

        print('Number of epochs:', n_epochs)
        # print("Number of steps per epoch:", num_steps)
        # print("Number of total steps:", num_steps * n_epochs)

        # move into the graph and start the model
        loss_history = []
        train_acc_vec = []
        valid_acc_vec = []
        step_history = []

        with tf.Session(graph=graph, config=tf.ConfigProto(
                use_per_session_threads=True,
                inter_op_parallelism_threads=n_cores,
                intra_op_parallelism_threads=n_cores)) as sess:

            sess.run(other_var["init"])

            best_performance = None
            best_performance_epoch = 0
            for step in range(n_epochs):
                # run the model (sess.run)
                # compute the optimizer, loss and train_prediction in the graph
                # save the last two as l and predictions

                # put thet data into TF form:
                feed_dict = {other_var["tf_X_seq"]: X_seq_train, other_var["tf_y"]: y_train,
                             other_var["tf_X_feat"]: X_feat_train,
                             other_var["tf_step_size"]: step_size}

                # run the optimizer
                # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/opt/python/training/external_optimizer.py#L115
                other_var["optimizer"].minimize(sess, feed_dict=feed_dict)
                l = sess.run(other_var["loss"], feed_dict=feed_dict)
                loss_history.append(l)  # keep storing the full loss history

                # sometimes print the actual training prediction (l)
                if (step % print_every == 0):
                    train_accuracy = self._accuracy_in_session(sess, other_var,
                                                               X_feat_train, X_seq_train, y_train)
                    valid_accuracy = self._accuracy_in_session(sess, other_var,
                                                               X_feat_valid, X_seq_valid, y_valid)
                    # append the prediction accuracies
                    train_acc_vec.append(train_accuracy)
                    valid_acc_vec.append(valid_accuracy)
                    step_history.append(step / num_steps)
                    print('Step %4d: loss %f, train mse: %f, validation mse: %f' %
                          (step, l, train_accuracy, valid_accuracy))
                    # check if this is the best accuracy
                    if best_performance is None or valid_accuracy <= best_performance:
                        best_performance = valid_accuracy
                        best_performance_epoch = step

                if early_stop_patience is not None and step > best_performance_epoch + early_stop_patience:
                    print("Early stopping. best_performance_epoch: %d, best_performance: %f" %
                          (best_performance_epoch, best_performance))
                    break

            # get the test accuracies
            train_accuracy_final = self._accuracy_in_session(sess, other_var,
                                                             X_feat_train, X_seq_train, y_train)
            valid_accuracy_final = self._accuracy_in_session(sess, other_var,
                                                             X_feat_valid, X_seq_valid, y_valid)
            print('Validation accuracy final: %f' % valid_accuracy_final)

            # store the fitted weights
            var_res = self._get_var_res_sess(sess, var)

            # store also the quasi splines fit

            if self._param["n_splines"] is not None:
                self._splines["quasi_X"] = [self._predict_in_session(sess, other_var,
                                                                     X_feat_train[i:(i + 1)],
                                                                     X_seq_train[i:(i + 1)],
                                                                     variable="spline_quasi_X")
                                            for i in range(X_feat_train.shape[0])]
                # transform into the appropriate form
                self._splines["quasi_X"] = np.concatenate([x[0][np.newaxis] for x in self._splines["quasi_X"]])

            accuracy = {
                "loss_history": np.array(loss_history),
                "step_history": np.array(step_history),
                "train_acc_history": np.array(train_acc_vec),
                "val_acc_history": np.array(valid_acc_vec),
                "train_acc_final": train_accuracy_final,
                "val_acc_final": valid_accuracy_final,
                "best_val_acc": best_performance,
                "best_val_acc_epoch": best_performance_epoch,
                "test_acc_final": None,  # test_accuracy_final,
                "y_test": None,  # y_test,
                "y_test_prediction": None,  # test_prediction.eval(),
                "id_vec_test": None  # id_vec_test
            }
            self._accuracy = accuracy

        toc = time.time()
        exec_time = toc - tic
        self._exec_time = exec_time
        print('That took %fs' % exec_time)
        # weights = {"motif_base_weights": motif_base_weights,
        #            "motif_weights": motif_weights,
        #            "motif_bias": motif_bias,
        #            "final_bias": final_bias,
        #            "feature_weights": feature_weights,
        #            "spline_pred": spline_pred
        #            }
        return var_res

    def _train_adam(self, X_feat_train, X_seq_train, y_train,
                    X_feat_valid, X_seq_valid, y_valid,
                    graph, var, other_var,
                    early_stop_patience=None,
                    n_cores=3):
        """
        Train the model actual model
        Updates weights / variables, computes and returns the training and validation accuracy
        """
        tic = time.time()
        # take out the parameters for conveience
        n_epochs = self._param["n_epochs"]
        batch_size = self._param["batch_size"]
        print_every = self._param["print_every"]
        step_size = self._param["step_size"]
        step_decay = self._param["step_decay"]
        step_epoch = self._param["step_epoch"]
        N_train = y_train.shape[0]
        num_steps = N_train // batch_size

        print('Number of epochs:', n_epochs)
        print("Number of steps per epoch:", num_steps)
        print("Number of total steps:", num_steps * n_epochs)

        # move into the graph and start the model
        loss_history = []
        train_acc_vec = []
        valid_acc_vec = []
        step_history = []

        with tf.Session(graph=graph, config=tf.ConfigProto(
                use_per_session_threads=True,
                inter_op_parallelism_threads=n_cores,
                intra_op_parallelism_threads=n_cores)) as sess:

            sess.run(other_var["init"])

            print('Initialized')
            epoch_count = 0
            best_performance = None
            best_performance_epoch = 0
            for step in range(num_steps * n_epochs):
                # where in the model are we
                # get the batch data + batch labels
                offset = (step * batch_size) % (N_train - batch_size)
                X_seq_batch = X_seq_train[offset:(offset + batch_size), :, :, :]
                X_feat_batch = X_feat_train[offset:(offset + batch_size), :]
                y_batch = y_train[offset:(offset + batch_size), :]

                # run the model (sess.run)
                # compute the optimizer, loss and train_prediction in the graph
                # save the last two as l and predictions

                epoch = step // num_steps
                if epoch_count < epoch:
                    step_size *= step_decay
                    epoch_count += step_epoch

                # put thet data into TF form:
                feed_dict = {other_var["tf_X_seq"]: X_seq_batch, other_var["tf_y"]: y_batch,
                             other_var["tf_X_feat"]: X_feat_batch,
                             other_var["tf_step_size"]: step_size}

                _, l = sess.run([other_var["optimizer"], other_var["loss"]], feed_dict=feed_dict)
                loss_history.append(l)
                # sometimes print the actual training prediction (l)
                # also access the variables from graph:
                # valid_prediction.eval()
                if (step % print_every == 0):
                    # append the prediction accuracies
                    # DONE: remove the constant evaluation code and feed_dict to do the task
                    train_accuracy = self._accuracy_in_session(sess, other_var,
                                                               X_feat_train, X_seq_train, y_train)
                    valid_accuracy = self._accuracy_in_session(sess, other_var,
                                                               X_feat_valid, X_seq_valid, y_valid)
                    train_acc_vec.append(train_accuracy)
                    valid_acc_vec.append(valid_accuracy)
                    step_history.append(step / num_steps)
                    print('Step %4d (epoch %d): loss %f, train mse: %f, validation mse: %f' %
                          (step, epoch, l, train_accuracy, valid_accuracy))

                    # check if this is the best accuracy
                    if best_performance is None or valid_accuracy <= best_performance:
                        best_performance = valid_accuracy
                        best_performance_epoch = epoch

                if early_stop_patience is not None and epoch > best_performance_epoch + early_stop_patience:
                    print("Early stopping. best_performance_epoch: %d, best_performance: %f" %
                          (best_performance_epoch, best_performance))
                    break

            # get the test accuracies
            train_accuracy_final = self._accuracy_in_session(sess, other_var,
                                                             X_feat_train, X_seq_train, y_train)
            valid_accuracy_final = self._accuracy_in_session(sess, other_var,
                                                             X_feat_valid, X_seq_valid, y_valid)
            print('Validation accuracy final: %f' % valid_accuracy_final)

            # store the fitted weights
            var_res = self._get_var_res_sess(sess, var)

            # store also the quasi splines fit

            if self._param["n_splines"] is not None:
                self._splines["quasi_X"] = [self._predict_in_session(sess, other_var, X_feat_train[i:(i + 1)], X_seq_train[i:(i + 1)],
                                                                     variable="spline_quasi_X") for i in range(X_feat_train.shape[0])]
                # transform into the appropriate form
                self._splines["quasi_X"] = np.concatenate([x[0][np.newaxis] for x in self._splines["quasi_X"]])

            accuracy = {
                "loss_history": np.array(loss_history),
                "step_history": np.array(step_history),
                "train_acc_history": np.array(train_acc_vec),
                "val_acc_history": np.array(valid_acc_vec),
                "train_acc_final": train_accuracy_final,
                "val_acc_final": valid_accuracy_final,
                "best_val_acc": best_performance,
                "best_val_acc_epoch": best_performance_epoch,
                "test_acc_final": None,  # test_accuracy_final,
                "y_test": None,  # y_test,
                "y_test_prediction": None,  # test_prediction.eval(),
                "id_vec_test": None  # id_vec_test
            }
            self._accuracy = accuracy

        toc = time.time()
        exec_time = toc - tic
        self._exec_time = exec_time
        print('That took %fs' % exec_time)
        # weights = {"motif_base_weights": motif_base_weights,
        #            "motif_weights": motif_weights,
        #            "motif_bias": motif_bias,
        #            "final_bias": final_bias,
        #            "feature_weights": feature_weights,
        #            "spline_pred": spline_pred
        #            }
        return var_res

    def predict(self, X_feat, X_seq):
        """
        Predict the response variable :py:attr:`y` for new input data (:py:attr:`X_feat`, :py:attr:`X_seq`).

        Args:
            X_feat: Feature design matrix. Same format as :py:attr:`X_feat` in :py:meth:`train`
            X_seq:  Sequenc design matrix. Same format as  :py:attr:`X_seq` in :py:meth:`train`
        """

        # insert one dimension - backcompatiblity
        X_seq = np.expand_dims(X_seq, axis=1)

        return self._get_other_var(X_feat, X_seq, variable="y_pred")

    def _get_other_var(self, X_feat, X_seq, variable="y_pred"):
        """
        Get the value of a variable from other_vars (from a tf-graph)
        """
        if self.is_trained() is False:
            print("Model not fitted yet. Use object.fit() to fit the model.")
            return

        # input check:
        assert X_seq.shape[0] == X_feat.shape[0]

        # TODO - check this
        # sequence can be wider or thinner?
        # assert self._param["seq_length"] == X_seq.shape[2]
        assert self._param["n_add_features"] == X_feat.shape[1]

        # setup graph and variables
        graph = tf.Graph()
        var = self._convert_to_var(graph, self._var_res)
        other_var = self._build_graph(graph, var)
        with tf.Session(graph=graph) as sess:
            sess.run(other_var["init"])
            # predict
            y = self._predict_in_session(sess, other_var, X_feat, X_seq, variable)

        return y

    def _test(self, X_feat_test, X_seq_test, y_test, id_vec=None):
        """
        Save the model's accuracy. It saves the test accuracy internally and returns the test accuracy
        """
        y_pred = self.predict(X_feat_test, X_seq_test)

        test_acc_final = ce.mse(y_pred, y_test)

        self._accuracy["test_acc_final"] = test_acc_final
        self._accuracy["y_test"] = y_test
        self._accuracy["y_test_prediction"] = y_pred
        self._accuracy["id_vec_test"] = id_vec

        # print("Model accuracy updated. Use model.get_accuracy() to obtain it.")

        return test_acc_final

    def print_weights(self):
        """
        Nicely print the fitted weights
        """
        return analyze.print_report(self.get_weights())

    def plot_accuracy(self):
        """
        Plot the accuracy history.
        """
        return analyze.plot_accuracy(self.get_accuracy())

    def plot_pos_bias(self):
        """
        Plot the positional bias.
        """
        if self._param["n_splines"] is None:
            print("Model was fitted without positional bias. n_splines has to be not None to use it")
            return

        return analyze.plot_pos_bias(self.get_weights())

    # save the object
    def to_dict(self):
        """
        Returns:
            dict: Concise represented as a dictionary.
        """
        final_res = {
            "param": self._param,
            "unused_param": self.unused_param,
            "execution_time": self._exec_time,
            "output": {"accuracy": self.get_accuracy(),
                       "weights": self.get_weights(),
                       "splines": self._splines
                       }
        }
        return final_res

    def save(self, file_path):
        """
        Save the object to a file in a .json format

        Args:
            file_path (str): Where to save the file.
        """
        helper.write_json(self.to_dict(), file_path)

    def _set_var_res(self, weights):
        """
        Transform the weights to var_res
        """
        if weights is None:
            return

        # layer 1
        motif_base_weights_raw = np.swapaxes(weights["motif_base_weights"], 2, 0)
        motif_base_weights = motif_base_weights_raw[np.newaxis]
        motif_bias = weights["motif_bias"]

        feature_weights = weights["feature_weights"]
        spline_weights = weights["spline_weights"]

        # filter
        motif_weights = weights["motif_weights"]
        final_bias = weights["final_bias"]

        var_res = {
            "motif_base_weights": motif_base_weights,
            "motif_bias": motif_bias,
            "spline_weights": spline_weights,
            "feature_weights": feature_weights,
            "motif_weights": motif_weights,
            "final_bias": final_bias
        }

        # cast everything to float32
        var_res = {key: value.astype(np.float32) if value is not None else None for key, value in var_res.items()}

        self._var_res = var_res

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Load the object from a dictionary (produced with :py:func:`Concise.to_dict`)

        Returns:
            Concise: Loaded Concise object.
        """

        # convert the output into a proper form
        obj_dict['output'] = helper.rec_dict_to_numpy_dict(obj_dict["output"])

        helper.dict_to_numpy_dict(obj_dict['output'])
        if "trained_global_model" in obj_dict.keys():
            raise Exception("Found trained_global_model feature in dictionary. Use ConciseCV.load to load this file.")
        dc = Concise(**obj_dict["param"])

        # touch the hidden arguments
        dc._param = obj_dict["param"]
        if obj_dict["output"]["weights"] is None:
            dc._model_fitted = False
        else:
            dc._model_fitted = True
            dc._exec_time = obj_dict["execution_time"]

        dc.unused_param = obj_dict["unused_param"]
        dc._accuracy = obj_dict["output"]["accuracy"]
        dc._splines = obj_dict["output"]["splines"]

        weights = obj_dict["output"]["weights"]

        if weights is not None:
            # fix the dimensionality of X_feat in case it was 0 dimensional
            if weights["feature_weights"].shape == (0,):
                weights["feature_weights"].shape = (0, obj_dict["param"]["num_tasks"])
            dc._set_var_res(weights)

        return dc

    @classmethod
    def load(cls, file_path):
        """
        Load the object from a JSON file (saved with :py:func:`Concise.save`).

        Returns:
            Concise: Loaded Concise object.
        """

        # convert back to numpy
        data = helper.read_json(file_path)
        return Concise.from_dict(data)


class ConciseCV(object):
    """Class for training the CONCISE in cross-validation

    Args:
        concise_model (Concise): Initialized Concise model with :py:func:`Concise`
    """

    # TODO: check the input - all sub-models have to have the same
    def __init__(self, concise_model):

        # TODO: extract the parameters

        self._concise_model = concise_model
        self._cv_model = None
        self._kf = None
        self._concise_global_model = None

    def get_param(self):
        """
        Returns:
            dict: Model's parameter list.
        """
        return self._param

    def get_unused_param(self):
        """
        Returns:
            dict: Model's additional parameters specified with :py:attr:`**kwargs` in :py:func:`Concise.__init__` of :py:attr:`concise_model`.
        """

        return self._concise_model.get_unused_param()

    # TODO: Check

    @staticmethod
    def _get_folds(n_rows, n_folds, use_stored):
        """
        Get the used CV folds
        """
        # n_folds = self._n_folds
        # use_stored = self._use_stored_folds
        # n_rows = self._n_rows

        if use_stored is not None:
            # path = '~/concise/data-offline/lw-pombe/cv_folds_5.json'
            with open(os.path.expanduser(use_stored)) as json_file:
                json_data = json.load(json_file)

            # check if we have the same number of rows and folds:
            if json_data['N_rows'] != n_rows:
                raise Exception('N_rows from folds doesnt match the number of rows of X_seq, X_feat, y')

            if json_data['N_folds'] != n_folds:
                raise Exception('n_folds dont match', json_data['N_folds'], n_folds)

            kf = [(np.array(train), np.array(test)) for (train, test) in json_data['folds']]
        else:
            kf = KFold(n_splits=n_folds).split(np.zeros((n_rows, 1)))

        # store in a list
        i = 1
        folds = []
        for train, test in kf:
            fold = "fold_" + str(i)
            folds.append((fold, train, test))
            i = i + 1
        return folds

    def get_folds(self):
        """
        Returns:
            dict: CV-fold indicies.
        """
        if self._kf is None:
            print("Model not trained yet. Train the model with model.train()")
            return
        else:
            return self._kf

    def train(self, X_feat, X_seq, y, id_vec=None, n_folds=10, use_stored_folds=None, n_cores=1,
              train_global_model=False):
        """Train the Concise model in cross-validation.

        Args:
            X_feat: See :py:func:`concise.Concise.train`
            X_seq: See :py:func:`concise.Concise.train`
            y: See :py:func:`concise.Concise.train`
            id_vec: List of character id's used to differentiate the trainig samples. Returned by :py:func:`concise.prepare_data`.
            n_folds (int): Number of CV-folds to use.
            use_stored_folds (chr or None): File path to a .json file containing the fold information (as returned by :py:func:`concise.ConciseCV.get_folds`). If None, the folds are generated.
            n_cores (int): Number of CPU cores used for training. If available, GPU is used for training and this argument is ignored.
            train_global_model (bool): In addition to training the model in cross-validation, should the global model be fitted (using all the samples from :code:`(X_feat, X_seq, y)`). 
        """
        # TODO: input check - dimensions
        self._use_stored_folds = use_stored_folds
        self._n_folds = n_folds
        self._n_rows = X_feat.shape[0]

    # TODO: - fix the get_cv_accuracy
    # save:
    # - each model
    # - each model's performance
    # - each model's predictions
    # - globally:
    #    - mean perfomance
    #    - sd performance
    #    - predictions

        self._kf = self._get_folds(self._n_rows, self._n_folds, self._use_stored_folds)

        cv_obj = {}

        if id_vec is None:
            id_vec = np.arange(1, self._n_rows + 1)

        best_val_acc_epoch_l = []
        for fold, train, test in self._kf:
            X_feat_train = X_feat[train]
            X_seq_train = X_seq[train]
            y_train = y[train]

            X_feat_test = X_feat[test]
            X_seq_test = X_seq[test]
            y_test = y[test]
            id_vec_test = id_vec[test]
            print(fold, "/", n_folds)

            # copy the object
            dc = copy.deepcopy(self._concise_model)
            dc.train(X_feat_train, X_seq_train, y_train,
                     X_feat_test, X_seq_test, y_test,
                     n_cores=n_cores
                     )

            dc._test(X_feat_test, X_seq_test, y_test, id_vec_test)
            cv_obj[fold] = dc
            best_val_acc_epoch_l.append(dc.get_accuracy()["best_val_acc_epoch"])
        self._cv_model = cv_obj

        # additionaly train the global model
        if train_global_model:
            dc = copy.deepcopy(self._concise_model)

            # overwrite n_epochs with the best average number of best epochs
            dc._param["n_epochs"] = int(np.array(best_val_acc_epoch_l).mean())
            print("tranining global model with n_epochs = " + str(dc._param["n_epochs"]))

            dc.train(X_feat, X_seq, y,
                     n_cores=n_cores
                     )
            dc._test(X_feat, X_seq, y, id_vec)
            self._concise_global_model = dc

    def get_CV_prediction(self):
        """
        Returns:
            np.ndarray: Predictions on the hold-out folds (unseen data, corresponds to :py:attr:`y`).
        """
        # TODO: get it from the test_prediction ...
        # test_id, prediction
        # sort by test_id
        predict_vec = np.zeros((self._n_rows, self._concise_model._num_tasks))
        for fold, train, test in self._kf:
            acc = self._cv_model[fold].get_accuracy()
            predict_vec[test, :] = acc["y_test_prediction"]
        return predict_vec

    def get_CV_accuracy(self):
        """
        Returns:
            float: Prediction accuracy in CV.
        """
        accuracy = {}
        for fold, train, test in self._kf:
            acc = self._cv_model[fold].get_accuracy()
            accuracy[fold] = acc["test_acc_final"]
        return accuracy

    def get_CV_models(self):
        """
        Returns:
            list of Concise: List of fitted models.
        """
        return self._cv_model

    def get_global_model(self):
        """
        Returns:
            Concise: Globally fitted model.
        """
        return self._concise_global_model

    def to_dict(self):
        """
        Returns:
            dict: ConciseCV represented as a dictionary.
        """
        param = {
            "n_folds": self._n_folds,
            "n_rows": self._n_rows,
            "use_stored_folds": self._use_stored_folds
        }

        if self._concise_global_model is None:
            trained_global_model = None
        else:
            trained_global_model = self._concise_global_model.to_dict()

        obj_dict = {"param": param,
                    "folds": self._kf,
                    "init_model": self._concise_model.to_dict(),
                    "trained_global_model": trained_global_model,
                    "output": {fold: model.to_dict() for fold, model in self.get_CV_models().items()}
                    }
        return obj_dict

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Load the object from a dictionary (produced with :py:func:`ConciseCV.to_dict`)
        Returns:
            ConciseCV: Loaded ConciseCV object.
        """
        default_model = Concise()
        cvdc = ConciseCV(default_model)
        cvdc._from_dict(obj_dict)
        return cvdc

    def _from_dict(self, obj_dict):
        """
        Initialize a model from the dictionary
        """
        self._n_folds = obj_dict["param"]["n_folds"]
        self._n_rows = obj_dict["param"]["n_rows"]
        self._use_stored_folds = obj_dict["param"]["use_stored_folds"]

        self._concise_model = Concise.from_dict(obj_dict["init_model"])

        if obj_dict["trained_global_model"] is None:
            self._concise_global_model = None
        else:
            self._concise_global_model = Concise.from_dict(obj_dict["trained_global_model"])

        self._kf = [(fold, np.asarray(train), np.asarray(test)) for fold, train, test in obj_dict["folds"]]
        self._cv_model = {fold: Concise.from_dict(model_dict) for fold, model_dict in obj_dict["output"].items()}

    def save(self, file_path):
        """
        Save the object to a file in a .json format

        Args:
            file_path (str): Where to save the file.
        """
        helper.write_json(self.to_dict(), file_path)

    @classmethod
    def load(cls, file_path):
        """
        Load the object from a JSON file (saved with :py:func:`ConciseCV.save`)

        Returns:
            ConciseCV: Loaded ConciseCV object.
        """
        data = helper.read_json(file_path)
        return ConciseCV.from_dict(data)
