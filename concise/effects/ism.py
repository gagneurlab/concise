# In-silico mutagenesis
import numpy as np
import pandas as pd




# Perhaps the diff type should be returned as a dict instead?!
def ism(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
        output_filter_mask = None, out_annotation= None, diff_type="log_odds", rc_handling = "maximum"):
    """
    :param model: Keras model
    :param ref: Reference sequence (1-hot)
    :param alt: Alternative sequence (1-hot)
    :param ref_rc: Reference sequence (1-hot)
    :param alt_rc: Alternative sequence (1-hot)
    :param mutation_positions: Position of the mutation(s) per sequence
    :param diff_type: Select difference type to return ["log_odds", "diff"]
    :param rc_handling: How should the fwd or rc prediction be selected ["average", "maximum"]
    :return: Will return the predicted effects in the same shape as the output shape of the model
    """

    seqs = {"ref":ref, "ref_rc":ref_rc, "alt":alt, "alt_rc":alt_rc}

    assert diff_type in ["log_odds", "diff"]
    assert rc_handling in ["average", "maximum"]
    assert np.all([np.array(ref.shape) == np.array(seqs[k].shape) for k in seqs.keys() if k != "ref"])
    assert ref.shape[0] == mutation_positions.shape[0]
    assert len(mutation_positions.shape)==1

    # determine which outputs should be selected
    if output_filter_mask is None:
        if out_annotation is None:
            output_filter_mask = np.arange(out_annotation_all_outputs.shape[0])
        else:
            output_filter_mask = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]

    # make sure the labels are assigned correctly
    out_annotation = out_annotation_all_outputs[output_filter_mask]

    preds = {}
    for k in seqs:
        #preds[k] = model.predict(seqs[k])
        preds[k] = np.array(model.predict(seqs[k])[..., output_filter_mask])

    if diff_type == "log_odds":
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
    elif diff_type == "diff":
        diffs = preds["alt"] - preds["ref"]
        diffs_rc = preds["alt_rc"] - preds["ref_rc"]

    if rc_handling == "average":
        diffs = np.mean([diffs, diffs_rc], axis = 0)
    elif rc_handling == "maximum":
        replace_filt = np.abs(diffs) < np.abs(diffs_rc)
        diffs[replace_filt] = diffs_rc[replace_filt]

    diffs = pd.DataFrame(diffs, columns=out_annotation)

    return {"ism": diffs}
