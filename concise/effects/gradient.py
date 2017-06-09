import numpy as np
from keras import backend as kB
import pandas as pd
from concise.effects.util import *
import copy

def predict_vals(input_data, mutated_positions, apply_function=None, output_concat_axis=0, batch_size=100, **kwargs):
    outputs = {}
    # if type(input_data) not in [list, tuple, dict]:
    #    input_data = [input_data]
    # assert(len(input_data)>0)
    # for el in input_data:
    #    assert(el.shape[0] == mutated_positions.shape[0])
    batch_idx = 0
    for batch_idx in range(int(np.ceil(mutated_positions.shape[0]/batch_size))):
        batched_input = get_batch(input_data, batch_size, batch_idx)
        start_idx = (batch_idx) * batch_size
        end_idx = min((batch_idx + 1) * batch_size, mutated_positions.shape[0])
        #if batched_input is None:
        #    break
        res = apply_function(input_data=batched_input, mutated_positions=mutated_positions[start_idx:end_idx,...], **kwargs)
        batch_idx += 1
        for k in res:
            if k not in outputs:
                outputs[k] = [res[k]]
            else:
                outputs[k].append(res[k])
    for k in outputs:
        outputs[k] = concatenate_by_input_type(input_data, outputs[k], output_concat_axis=output_concat_axis)
    return outputs

def concatenate_by_input_type(input_data, to_concat, output_concat_axis=0):
    if isinstance(input_data, (list, tuple)):
        out_obj = []
        for x in range(len(input_data)):
            concat_els = []
            for el in to_concat:
                concat_els.append(el[x])
            out_obj.append(np.concatenate(concat_els, axis=output_concat_axis))
        if isinstance(input_data, tuple):
            out_obj = tuple(out_obj)
        return out_obj
    elif isinstance(input_data, dict):
        out_obj = {}
        for k in input_data:
            concat_els = []
            for el in to_concat:
                concat_els.append(el[k])
            out_obj[k] = np.concatenate(concat_els, axis=output_concat_axis)
        return out_obj
    elif isinstance(input_data, np.ndarray):
        return np.concatenate(to_concat, axis=output_concat_axis)
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")


def get_batch(input_data, batchsize, batch_idx):
    # yield the output object
    return_obj = None
    start_idx = (batch_idx) * batchsize
    if isinstance(input_data, (list, tuple)):
        out_obj = []
        for x in range(len(input_data)):
            end_idx = min((batch_idx + 1) * batchsize, input_data[x].shape[0])
            if start_idx > end_idx:
                return None
            out_obj.append(input_data[x][start_idx:end_idx, ...])
        if isinstance(input_data, tuple):
            out_obj = tuple(out_obj)
        return_obj = out_obj
    elif isinstance(input_data, dict):
        out_obj = {}
        for k in input_data:
            end_idx = min((batch_idx + 1) * batchsize, input_data[k].shape[0])
            if start_idx > end_idx:
                return None
            out_obj[k] = (input_data[k][start_idx:end_idx, ...])
        return_obj = out_obj
    elif isinstance(input_data, np.ndarray):
        end_idx = min((batch_idx + 1) * batchsize, input_data.shape[0])
        if start_idx > end_idx:
            return None
        out_obj = (input_data[start_idx:end_idx, ...])
        return_obj = out_obj
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")
    return return_obj

def general_diff(left, right):
    if isinstance(left, (list, tuple)):
        out = []
        for l, r in zip(left, right):
            out.append(l - r)
        return out
    elif isinstance(left, dict):
        out = {}
        for k in left:
            out[k] = left[k] - right[k]
        return out
    elif isinstance(left, np.ndarray):
        return left - right
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

def general_sel(remains, return_if_smaller_than):
    # Generalisation of: sel = np.abs(diff_fwd) < np.abs(diff_rc)
    if isinstance(remains, (list, tuple)):
        return [np.abs(rem) < np.abs(test) for rem, test in zip(remains, return_if_smaller_than)]
    elif isinstance(remains, dict):
        out = {}
        for k in remains:
            out[k] = np.abs(remains[k]) < np.abs(return_if_smaller_than[k])
        return out
    elif isinstance(remains, np.ndarray):
        return np.abs(remains) < np.abs(return_if_smaller_than)
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

def replace_by_sel(to_be_edited, alt_values, sel):
    if isinstance(to_be_edited, (list, tuple)):
        for t, a, s in zip(to_be_edited, alt_values, sel):
            t[s] = a[s]
    elif isinstance(to_be_edited, dict):
        for k in to_be_edited:
            to_be_edited[k][sel[k]] = alt_values[k][sel[k]]
    elif isinstance(to_be_edited, np.ndarray):
        to_be_edited[sel] = alt_values[sel]
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")


def input_times_grad(input, gradient, positions):
    def multiply_input_grad(grad, inp, positions):
        if positions.shape[0] != grad.shape[0]:
            raise Exception("At the moment exactly one (mutational) position is allowed per input sequence!")

        if not np.all(np.array(grad.shape) == np.array(inp.shape)):
            raise Exception("Input sequence and gradient have to have the same dimensions!")

        scores = grad[range(positions.shape[0]), positions, :]
        input_at_mut = inp[range(positions.shape[0]), positions, :]

        # Calculate gradient * input
        return (scores * input_at_mut).sum(axis=1)

    assert (len(positions.shape) == 1)  # has to be 1-dimensional
    positions = positions.astype(np.int)

    if type(input) is not type(gradient):
        raise Exception("Input sequence and gradient have to be the same type!")

    if isinstance(input, (list, tuple)):
        if not (len(input) == len(gradient)):
            raise Exception("Internal Error: Input and gradient list objects have different lenghts!")
        out_obj = []
        for x in range(len(input)):
            out_obj.append(multiply_input_grad(input[x], gradient[x], positions))
    elif isinstance(input, dict):
        if not np.all(np.in1d(input.keys(), gradient.keys())) or (len(input) != len(gradient)):
            raise Exception("Internal Error: Input and gradient dict objects have different keys!")
        out_obj = {}
        for k in input:
            out_obj[k] = multiply_input_grad(input[k], gradient[k], positions)
    elif isinstance(input, np.ndarray):
        out_obj = multiply_input_grad(input, gradient, positions)
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

    return out_obj


def __get_direct_saliencies__(input_data, score_func, mutated_positions, model):
    all_scores = {}
    method_name = "dGrad"
    # Take first element as it is the one with gradients
    if isinstance(input_data, list):
        input = [el for el in input_data]
    else:
        input = [input_data]
    if kB._BACKEND == "theano":
        if model.output._uses_learning_phase:
            input.append(0)
    else:
        input.append(0)
    scores = score_func(input)  # test phase, so learning_phase = 0
    if isinstance(input_data, np.ndarray):
        scores = scores[0]
    scores = input_times_grad(input_data, scores, mutated_positions)
    all_scores[method_name] = scores
    return all_scores


def __generate_direct_saliency_functions__(model, out_annotation_all_outputs, out_annotation=None):
    sal_funcs = {}
    if out_annotation is not None:
        sel_outputs = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]
    else:
        sel_outputs = np.arange(out_annotation_all_outputs.shape[0])
    for i in sel_outputs:
        inp = copy.copy(model.inputs)
        outp = model.layers[-1].output
        max_outp = outp[:, i]
        if kB._BACKEND == "theano":
            saliency = kB.gradients(max_outp.sum(), inp)
            if model.output._uses_learning_phase:
                inp.append(kB.learning_phase())
        else:
            saliency = kB.gradients(max_outp, inp)
            inp.append(kB.learning_phase())
        sal_funcs[out_annotation_all_outputs[i]] = kB.function(inp, saliency)
    return sal_funcs


def __generate_direct_saliency_functions_DEPRECATED__(model, out_annotation_all_outputs, out_annotation=None):
    sal_funcs = {}
    if out_annotation is not None:
        sel_outputs = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]
    else:
        sel_outputs = np.arange(out_annotation_all_outputs.shape[0])
    for i in sel_outputs:
        inp = model.layers[0].input
        outp = model.layers[-1].output
        max_outp = outp[:, i]
        saliency = kB.gradients(max_outp, inp)
        sal_funcs[out_annotation_all_outputs[i]] = kB.function([inp, kB.learning_phase()], saliency)
    return sal_funcs


# The function called from outside
def gradient_pred(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
                  output_filter_mask=None, out_annotation=None):
    """Gradient-based (saliency) variant effect prediction

    Based on the idea of [saliency maps](https://arxiv.org/pdf/1312.6034.pdf) the gradient-based prediction of
    variant effects uses the `gradient` function of the Keras backend to estimate the importance of a variant
    for a given output. This value is then multiplied by the input, as recommended by
    [Shrikumar et al., 2017](https://arxiv.org/pdf/1605.01713.pdf).

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

    # Returns

    Dictionary with three different entries:

    - ref: Gradient * input at the mutation position using the reference sequence.
        Forward or reverse-complement sequence is chose based on sequence direction caused
        the bigger absolute difference ('diff')
    - alt: Gradient * input at the mutation position using the alternative sequence. Forward or
        reverse-complement sequence is chose based on sequence direction caused the bigger
        absolute difference ('diff')
    - diff: 'alt' - 'ref'. Forward or reverse-complement sequence is chose based on sequence
        direction caused the bigger absolute difference.
    """
    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}

    for k in seqs:
        if not isinstance(seqs[k], (list, tuple, np.ndarray)):
            raise Exception("At the moment only models with list, tuple or np.ndarray inputs are supported.")

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

    # Generate the necessary gradient functions
    sal_funcs = __generate_direct_saliency_functions__(model, out_annotation_all_outputs, out_annotation)

    # ANALOGOUS TO ISM:
    # predict
    preds = {}

    for k in seqs:
        preds[k] = {}
        if "_rc" in k:
            mutated_positions_here = get_seq_len(ref)[1] - 1 - mutation_positions
        else:
            mutated_positions_here = mutation_positions
        for l in out_annotation:
            preds[k][l] = predict_vals(input_data=seqs[k], apply_function=__get_direct_saliencies__,
                                       score_func=sal_funcs[l], mutated_positions=mutated_positions_here, model = model)

    diff_ret_dGrad = {}
    pred_out = {"ref": {}, "alt": {}}
    for k in preds["ref"]:
        # TODO make list (and dict)-ready
        diff_fwd = general_diff(preds["alt"][k]["dGrad"],  preds["ref"][k]["dGrad"])
        diff_rc = general_diff(preds["alt_rc"][k]["dGrad"],  preds["ref_rc"][k]["dGrad"])
        sel = general_sel(diff_fwd, diff_rc)
        replace_by_sel(diff_fwd, diff_rc, sel)
        diff_ret_dGrad[k] = diff_fwd
        # Overwrite the fwd values with rc values if rc was selected
        replace_by_sel(preds["ref"][k]["dGrad"], preds["ref_rc"][k]["dGrad"], sel)
        replace_by_sel(preds["alt"][k]["dGrad"], preds["alt_rc"][k]["dGrad"], sel)
        pred_out["ref"][k] = preds["ref"][k]["dGrad"]
        pred_out["alt"][k] = preds["alt"][k]["dGrad"]

    return {"diff": pd.DataFrame(diff_ret_dGrad),
            "ref": pd.DataFrame(pred_out["ref"]),
            "alt": pd.DataFrame(pred_out["alt"])}




