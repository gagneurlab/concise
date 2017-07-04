import keras
import numpy as np
import pandas as pd
from scipy.stats.stats import ttest_ind
from scipy.special import logit
from concise.effects.util import *


# TODO: Move to layer definition:
from keras.layers.core import Dropout
import keras.backend as K
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


def replace_dict_values(in_dict, from_value, to_value):
    out_dict = {}
    for k in in_dict:
        if isinstance(in_dict[k], dict):
            out_dict[k] = replace_dict_values(in_dict[k], from_value, to_value)
        elif isinstance(in_dict[k], list):
            out_dict[k] = []
            for el in in_dict[k]:
                if isinstance(el, dict):
                    out_dict[k].append(replace_dict_values(el, from_value, to_value))
                else:
                    if el == from_value:
                        out_dict[k].append(to_value)
                    else:
                        out_dict[k].append(el)
        else:
            if in_dict[k] == from_value:
                out_dict[k] = to_value
            else:
                out_dict[k] = in_dict[k]
    return out_dict


def pred_do(model, input_data, output_filter_mask, dropout_iterations):
    diff_outputs = []
    for k in range(dropout_iterations):
        # diff_outputs.append(np.array(self.predict_vals(input_data)[0]))
        diff_outputs.append(np.array(model.predict(input_data)[..., output_filter_mask]))
    return np.array(diff_outputs)


def subset_array_by_index(arr, idx):
    assert (np.all(np.array(arr.shape[1:]) == np.array(idx.shape[1:])))
    assert (arr.shape[0] / 2 == idx.shape[0])
    pred_out_sel = []
    for c in range(arr.shape[1]):
        sel_idxs = np.arange(arr.shape[0] / 2) * 2 + idx[:, c]
        pred_out_sel.append(arr[sel_idxs, c])
    return np.array(pred_out_sel).T


def overwite_by(main_arr, alt_arr, idx):
    assert (np.all(np.array(main_arr.shape[1:]) == np.array(idx.shape[1:])))
    assert (main_arr.shape[0] == idx.shape[0])
    assert (alt_arr.shape[0] == idx.shape[0])
    for c in range(main_arr.shape[1]):
        main_arr[idx[:, c], c] = alt_arr[idx[:, c], c]
    return main_arr


def test_overwite_by():
    a = np.array([[1, 2], [4, 5]])
    b = np.array([[1, 8], [4, 5]])
    overwite_by(a, b, a < b)
    assert(a[0, 1] == 8)

def get_range(input_data):
    vals = {"max": [], "min": []}

    def add_vals(new_vals):
        for k2 in new_vals:
            vals[k2].append(new_vals[k2])

    if isinstance(input_data, (list, tuple)):
        for el in input_data:
            add_vals(get_range(el))
    elif isinstance(input_data, dict):
        for k in input_data:
            add_vals(get_range(input_data[k]))
    elif isinstance(input_data, np.ndarray):
        return {"max": input_data.max(), "min": input_data.min()}
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")
    vals["max"] = max(vals["max"])
    vals["min"] = min(vals["min"])
    return vals

def apply_over_single(input_data, apply_func, select_return_elm=None, **kwargs):
    if isinstance(input_data, (list, tuple)):
        return [apply_over_single(el, apply_func, select_return_elm, **kwargs) for el in input_data]
    elif isinstance(input_data, dict):
        out = {}
        for k in input_data:
            out[k] = apply_over_single(input_data[k], apply_func, select_return_elm, **kwargs)
        return out
    elif isinstance(input_data, np.ndarray):
        ret = apply_func(input_data, **kwargs)
        if select_return_elm is not None:
            ret = ret[select_return_elm]
        return ret
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

def apply_over_double(input_data_a, input_data_b, apply_func, select_return_elm=None, **kwargs):
    if isinstance(input_data_a, (list, tuple)):
        return [apply_over_double(el_a, el_b, apply_func, select_return_elm, **kwargs) for el_a, el_b in
                zip(input_data_a, input_data_b)]
    elif isinstance(input_data_a, dict):
        out = {}
        for k in input_data_a:
            out[k] = apply_over_double(input_data_a[k], input_data_b[k], apply_func, select_return_elm, **kwargs)
        return out
    elif isinstance(input_data_a, np.ndarray):
        ret = apply_func(input_data_a, input_data_b, **kwargs)
        if select_return_elm is not None:
            ret = ret[select_return_elm]
        return ret
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")


# The function called from outside
def dropout_pred(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
                 output_filter_mask=None, out_annotation=None, dropout_iterations=30):
    """Dropout-based variant effect prediction

        This method is based on the ideas in [Gal et al.](https://arxiv.org/pdf/1506.02142.pdf) where dropout
        layers are also actived in the model prediction phase in order to estimate model uncertainty. The
        advantage of this method is that instead of a point estimate of the model output the distribution of
        the model output is estimated.

        # Arguments
            model: Keras model
            ref: Input sequence with the reference genotype in the mutation position
            ref_rc: Reverse complement of the 'ref' argument
            alt: Input sequence with the alternative genotype in the mutation position
            alt_rc: Reverse complement of the 'alt' argument
            mutation_positions: Position on which the mutation was placed in the forward sequences
            out_annotation_all_outputs: Output labels of the model.
            output_filter_mask: Mask of boolean values indicating which model outputs should be used.
                Use this or 'out_annotation'
            out_annotation: List of outputs labels for which of the outputs (in case of a multi-task model) the
                predictions should be calculated.
            dropout_iterations: Number of prediction iterations to be performed in order to estimate the
                output distribution. Values greater than 30 are recommended to get a reliable p-value.

        # Returns

            Dictionary with a set of measures of the model uncertainty in the variant position. The ones of interest are:

            - do_{ref, alt}_mean: Mean of the model predictions given the respective input sequence and dropout.
                - Forward or reverse-complement sequences are chosen as for 'do_pv'.
            - do_{ref, alt}_var: Variance of the model predictions given the respective input sequence and dropout.
                - Forward or reverse-complement sequences are chosen as for 'do_pv'.
            - do_diff: 'do_alt_mean' - 'do_alt_mean', which is an estimate similar to ISM using diff_type "diff".
            - do_pv: P-value of a paired t-test, comparing the predictions of ref with the ones of alt. Forward or
                - reverse-complement sequences are chosen based on which pair has the lower p-value.
        """
    prefix = "do"

    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}

    assert np.all([np.array(get_seq_len(ref)) == np.array(get_seq_len(seqs[k])) for k in seqs.keys() if k != "ref"])
    assert get_seq_len(ref)[0] == mutation_positions.shape[0]
    assert len(mutation_positions.shape) == 1

    # determine which outputs should be selected
    if output_filter_mask is None:
        if out_annotation is None:
            output_filter_mask = np.arange(out_annotation_all_outputs.shape[0])
        else:
            output_filter_mask = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]

    # make sure the labels are assigned correctly
    out_annotation = out_annotation_all_outputs[output_filter_mask]

    # Instead of loading the model from a json file I will transfer the model architecture + weights in memory
    model_config = model._updated_config()
    alt_config = replace_dict_values(model_config, u"Dropout", u"BiDropout")

    # Custom objects have to be added before correctly!
    alt_model = keras.layers.deserialize(alt_config)

    # Transfer weights and biases
    alt_model.set_weights(model.get_weights())

    # ANALOGOUS TO ISM:
    # predict
    preds = {}
    for k in seqs:
        preds[k] = pred_do(alt_model, seqs[k], output_filter_mask=output_filter_mask, dropout_iterations=dropout_iterations)

    t, prob = ttest_ind(preds["ref"], preds["alt"], axis=0)
    t_rc, prob_rc = ttest_ind(preds["ref_rc"], preds["alt_rc"], axis=0)

    logit_prob = None
    logit_prob_rc = None
    pred_range = get_range(preds)
    # In case the predictions are bound to [0,1] it might make sense to use logit on the data, as the model output
    # could be probalilities
    if np.all([(pred_range[k] >= 0) and (pred_range[k] <= 1) for k in pred_range]):
        logit_preds = apply_over_single(preds, logit)
        logit_prob = apply_over_double(logit_preds["ref"], logit_preds["alt"], apply_func=ttest_ind,
                                       select_return_elm=1, axis=0)
        logit_prob_rc = apply_over_double(logit_preds["ref_rc"], logit_preds["alt_rc"], apply_func=ttest_ind,
                                       select_return_elm=1, axis=0)
    # fwd and rc are independent here... so this can be done differently here...

    sel = (np.abs(prob) > np.abs(prob_rc)).astype(np.int)  # Select the LOWER p-value among fwd and rc

    out_dict = {}

    out_dict["%s_pv" % prefix] = pd.DataFrame(overwite_by(prob, prob_rc, sel), columns=out_annotation)

    if logit_prob is not None:
        logit_sel = (np.abs(logit_prob) > np.abs(logit_prob_rc)).astype(np.int)
        out_dict["%s_logit_pv" % prefix] = pd.DataFrame(overwite_by(logit_prob, logit_prob_rc, logit_sel), columns=out_annotation)

    pred_means = {}
    pred_vars = {}
    pred_cvar2 = {}
    for k in preds:
        pred_means[k] = np.mean(preds[k], axis=0)
        pred_vars[k] = np.var(preds[k], axis=0)
        pred_cvar2[k] = pred_vars[k] / (pred_means[k] ** 2)

    mean_cvar = np.sqrt((pred_cvar2["ref"] + pred_cvar2["alt"]) / 2)
    mean_cvar_rc = np.sqrt((pred_cvar2["ref_rc"] + pred_cvar2["alt_rc"]) / 2)

    mean_cvar = overwite_by(mean_cvar, mean_cvar_rc, sel)
    ref_mean = overwite_by(pred_means["ref"], pred_means["ref_rc"], sel)
    alt_mean = overwite_by(pred_means["alt"], pred_means["alt_rc"], sel)
    ref_var = overwite_by(pred_vars["ref"], pred_vars["ref_rc"], sel)
    alt_var = overwite_by(pred_vars["alt"], pred_vars["alt_rc"], sel)

    out_dict["%s_ref_mean" % prefix] = pd.DataFrame(ref_mean, columns=out_annotation)
    out_dict["%s_alt_mean" % prefix] = pd.DataFrame(alt_mean, columns=out_annotation)

    out_dict["%s_ref_var" % prefix] = pd.DataFrame(ref_var, columns=out_annotation)
    out_dict["%s_alt_var" % prefix] = pd.DataFrame(alt_var, columns=out_annotation)

    out_dict["%s_cvar" % prefix] = pd.DataFrame(mean_cvar, columns=out_annotation)

    out_dict["%s_diff" % prefix] = out_dict["%s_alt_mean" % prefix] - out_dict["%s_ref_mean" % prefix]

    return out_dict



