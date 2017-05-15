import numpy as np
from  keras import backend as kB
import progressbar
import pandas as pd

#### The "personalised" version is reverted back to just checking that the input has the correct dimensions and then
#### the predicitons are made.

## TODO: Handle the output labelling of the model coherently and remove dependencies on output capture and IO_Sequence

def predict_vals(input, mutated_positions, apply_function=None, output_concat_axis = 0, **kwargs):
    outputs = {}
    assert(type(input) is list)
    assert(len(input)>0)
    for el in input:
        assert(el.shape[0] == mutated_positions.shape[0])
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
        if positions.shape[0] != gradient[idx].shape[0]:
            raise Exception("At the moment exactly on (mutational) position is allowed per input sequence!")

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

    if isinstance(obj, (list, tuple)):
        if not (len(input) == len(gradient)):
            raise Exception("Internal Error: Input and gradient list objects have different lenghts!")
        out_obj = []
        for x in range(len(input)):
            out_obj.append(multiply_input_grad(input[x], gradient[x],positions))
    elif isinstance(obj, dict):
        if not np.all(np.in1d(input.keys(), gradient.keys())) or (len(input) != len(gradient)):
            raise Exception("Internal Error: Input and gradient dict objects have different keys!")
        out_obj = {}
        for k in input:
            out_obj[k] = multiply_input_grad(input[k], gradient[k],positions)
    elif isinstance(obj, np.ndarray):
        out_obj = multiply_input_grad(input, gradient, positions)
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

    return out_obj

def __get_direct_saliencies__(input_data, score_func , mutated_positions):
    from collections import OrderedDict
    all_scores = {}
    method_name = "dGrad"
    import pdb
    # Take first element as it is the one with gradients
    scores = score_func([input_data])[0]
    scores = input_times_grad(input_data, scores, mutated_positions)
    all_scores[method_name] = scores
    return all_scores

def __generate_direct_saliency_functions__(model, out_annotation_all_outputs, out_annotation = None):
    sal_funcs = {}
    bar = progressbar.ProgressBar(widget_kwargs=dict(fill='.'))
    if out_annotation is not None:
        sel_outputs = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]
    else:
        sel_outputs = np.arange(out_annotation_all_outputs.shape[0])
    for i in bar(sel_outputs):
        inp = model.layers[0].input
        outp = model.layers[-1].output
        max_outp = outp[:,i]
        saliency = kB.gradients(max_outp, inp)
        sal_funcs[out_annotation_all_outputs[i]] = kB.function([inp], saliency)
    return sal_funcs

"""
class saliency_calcs(class_preds):
    def __init__(self, model_path):
        super(saliency_calcs, self).__init__(model_path + "/a.json", model_path + "/best_w.h5", model_path + "/pred.hdf5")
        self.num_output_tasks = self.output_labels.shape[0]
        self.sal_funcs = None
    #
    def __generate_direct_saliency_functions__(self):
        if self.sal_funcs is None:
            self.sal_funcs = []
            bar = progressbar.ProgressBar(widget_kwargs=dict(fill='.'))
            for i in bar(range(self.num_output_tasks)):
                inp = self.model.layers[0].input
                outp = self.model.layers[-1].output
                max_outp = outp[:,i]
                saliency = kB.gradients(max_outp, inp)
                self.sal_funcs.append(kB.function([inp], saliency))
    #
    def __get_direct_saliencies__(self, input_data, output_task_id , mutated_positions):
        from collections import OrderedDict
        all_scores = {}
        method_name = "dGrad"
        import pdb
        #pdb.set_trace()
        score_func = self.sal_funcs[output_task_id]
        # Take first element as it is the one with gradients
        scores = score_func([input_data])[0]
        scores = input_times_grad(input_data, scores, mutated_positions)
        all_scores[method_name] = scores
        return all_scores
    #
    def get_direct_saliencies(self, input_data, mutated_positions, selected_outputs = []):
        self.__generate_direct_saliency_functions__()
        output_labels = self.output_labels
        output_mask = range(self.num_output_tasks)
        if len(selected_outputs) != 0:
            output_mask, output_labels = self.__get_output_filter_mask__(selected_outputs)
            output_mask = np.where(output_mask)[0].tolist()
        # Dict with keys: gradient calculation type (e.g.: "dGrad"), values: list of scores with the order of output_labels
        outs = {}
        for out_task_idx in output_mask:
            scores = predict_vals(input=input_data, apply_function=self.__get_direct_saliencies__, output_task_id = out_task_idx, mutated_positions = mutated_positions)
            for k in scores:
                if len(scores[k].shape)==1:
                    if k not in outs:
                        outs[k] = [scores[k]]
                    else:
                        outs[k].append(scores[k])
                elif len(scores[k].shape)==2:
                    if k not in outs:
                        outs[k] = [scores[k][:,None,:]]
                    else:
                        outs[k].append(scores[k][:,None,:])
        #TODO: Is it desired to get a DataFrame object back?
        # Transform in to a labelled dataframe if possible
        for k in outs:
            if len(outs[k]) > 0:
                if len(outs[k][0].shape)==1:
                    outs[k] = np.array(outs[k]).T # The output labels should be columns
                    outs[k] = pd.DataFrame(outs[k], columns = output_labels)
                elif len(outs[k][0].shape)==3:
                    outs[k] = np.concatenate(outs[k], axis=1)
        return outs

"""



# The function called from outside
def gradient_pred(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
        output_filter_mask=None, out_annotation=None):

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
    #pdb.set_trace()

    for k in seqs:
        preds[k] = {}
        if "_rc" in k:
            mutated_positions_here = ref.shape[1]-1-mutation_positions
        else:
            mutated_positions_here = mutation_positions
        for l in out_annotation:
            preds[k][l] = predict_vals(input=seqs[k], apply_function=__get_direct_saliencies__,
                                       score_func=sal_funcs[l], mutated_positions=mutated_positions_here)



    diff_ret_dGrad = {}
    for k in preds["ref"]:
        diff_fwd = preds["alt"][k] - preds["ref"][k]
        diff_rc = preds["alt_rc"][k] - preds["ref_rc"][k]
        sel = np.abs(diff_fwd) < np.abs(diff_rc)
        diff_fwd[sel] = diff_rc[sel]
        diff_ret_dGrad[k] = diff_fwd
        # Overwrite the fwd values with rc values if rc was selected
        preds["ref"][k][sel] = preds["ref_rc"][k][sel]
        preds["alt"][k][sel] = preds["alt_rc"][k][sel]


    return {"diff": pd.DataFrame(diff_ret_dGrad[k]), "ref": pd.DataFrame(preds["ref"]), "alt": pd.DataFrame(preds["alt"])}

