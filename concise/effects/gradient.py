import numpy as np
from  keras import backend as kB
import pandas as pd


def predict_vals(input, mutated_positions, apply_function=None, output_concat_axis = 0, **kwargs):
    outputs = {}
    #if type(input) not in [list, tuple, dict]:
    #    input = [input]
    #assert(len(input)>0)
    #for el in input:
    #    assert(el.shape[0] == mutated_positions.shape[0])
    res = apply_function(input_data = input, mutated_positions = mutated_positions, **kwargs)
    for k in res:
        if k not in outputs:
            outputs[k] = [res[k]]
        else:
            outputs[k].append(res[k])
    for k in outputs:
        outputs[k] = np.concatenate(outputs[k], axis = output_concat_axis)
    return outputs


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

    import pdb
    #pdb.set_trace()

    if type(input) is not type(gradient):
        raise Exception("Input sequence and gradient have to be the same type!")

    if isinstance(input, (list, tuple)):
        if not (len(input) == len(gradient)):
            raise Exception("Internal Error: Input and gradient list objects have different lenghts!")
        out_obj = []
        for x in range(len(input)):
            out_obj.append(multiply_input_grad(input[x], gradient[x],positions))
    elif isinstance(input, dict):
        if not np.all(np.in1d(input.keys(), gradient.keys())) or (len(input) != len(gradient)):
            raise Exception("Internal Error: Input and gradient dict objects have different keys!")
        out_obj = {}
        for k in input:
            out_obj[k] = multiply_input_grad(input[k], gradient[k],positions)
    elif isinstance(input, np.ndarray):
        out_obj = multiply_input_grad(input, gradient, positions)
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

    return out_obj

def __get_direct_saliencies__(input_data, score_func , mutated_positions):
    all_scores = {}
    method_name = "dGrad"
    import pdb
    # Take first element as it is the one with gradients
    scores = score_func([input_data, 0])[0] # test phase, so learning_phase = 0
    scores = input_times_grad(input_data, scores, mutated_positions)
    all_scores[method_name] = scores
    return all_scores

def __generate_direct_saliency_functions__(model, out_annotation_all_outputs, out_annotation = None):
    sal_funcs = {}
    if out_annotation is not None:
        sel_outputs = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]
    else:
        sel_outputs = np.arange(out_annotation_all_outputs.shape[0])
    for i in sel_outputs:
        inp = model.layers[0].input
        outp = model.layers[-1].output
        max_outp = outp[:,i]
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
            ref: Gradient * input at the mutation position using the reference sequence.
                Forward or reverse-complement sequence is chose based on sequence direction caused
                the bigger absolute difference ('diff')
            alt: Gradient * input at the mutation position using the alternative sequence. Forward or
                reverse-complement sequence is chose based on sequence direction caused the bigger
                absolute difference ('diff')
            diff: 'alt' - 'ref'. Forward or reverse-complement sequence is chose based on sequence
                direction caused the bigger absolute difference.
    """
    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}

    assert np.all([np.array(ref.shape) == np.array(seqs[k].shape) for k in seqs.keys() if k != "ref"])
    assert ref.shape[0] == mutation_positions.shape[0]
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

    import pdb

    for k in seqs:
        preds[k] = {}
        if "_rc" in k:
            mutated_positions_here = ref.shape[1]-1-mutation_positions
        else:
            mutated_positions_here = mutation_positions
        for l in out_annotation:
            preds[k][l] = predict_vals(input=seqs[k], apply_function=__get_direct_saliencies__,
                                       score_func=sal_funcs[l], mutated_positions=mutated_positions_here)

    #pdb.set_trace()
    diff_ret_dGrad = {}
    pred_out = {"ref":{}, "alt":{}}
    for k in preds["ref"]:
        diff_fwd = preds["alt"][k]["dGrad"] - preds["ref"][k]["dGrad"]
        diff_rc = preds["alt_rc"][k]["dGrad"] - preds["ref_rc"][k]["dGrad"]
        sel = np.abs(diff_fwd) < np.abs(diff_rc)
        diff_fwd[sel] = diff_rc[sel]
        diff_ret_dGrad[k] = diff_fwd
        # Overwrite the fwd values with rc values if rc was selected
        preds["ref"][k]["dGrad"][sel] = preds["ref_rc"][k]["dGrad"][sel]
        preds["alt"][k]["dGrad"][sel] = preds["alt_rc"][k]["dGrad"][sel]
        pred_out["ref"][k] = preds["ref"][k]["dGrad"]
        pred_out["alt"][k] = preds["alt"][k]["dGrad"]


    return {"diff": pd.DataFrame(diff_ret_dGrad[k], columns = out_annotation),
            "ref": pd.DataFrame(pred_out["ref"], columns = out_annotation),
            "alt": pd.DataFrame(pred_out["alt"], columns = out_annotation)}

