# In-silico mutagenesis
import numpy as np
import pandas as pd
import warnings
from concise.effects.util import *


# Perhaps the diff type should be returned as a dict instead?!
def ism(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
        output_filter_mask=None, out_annotation=None, diff_type="log_odds", rc_handling="maximum"):
    """In-silico mutagenesis

    Using ISM in with diff_type 'log_odds' and rc_handling 'maximum' will produce predictions as used
    in [DeepSEA](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html). ISM offers two ways to
    calculate the difference between the outputs created by reference and alternative sequence and two
    different methods to select whether to use the output generated from the forward or from the
    reverse-complement sequences. To calculate "e-values" as mentioned in DeepSEA the same ISM prediction
    has to be performed on a randomised set of 1 million 1000genomes, MAF-matched variants to get a
    background of predicted effects of random SNPs.

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
        diff_type: "log_odds" or "diff". When set to 'log_odds' calculate scores based on log_odds, which assumes
            the model output is a probability. When set to 'diff' the model output for 'ref' is subtracted
            from 'alt'. Using 'log_odds' with outputs that are not in the range [0,1] nan will be returned.
        rc_handling: "average" or "maximum". Either average over the predictions derived from forward and
            reverse-complement predictions ('average') or pick the prediction with the bigger absolute
            value ('maximum').

    # Returns
        Dictionary with the key `ism` which contains a pandas DataFrame containing the calculated values
            for each (selected) model output and input sequence
    """

    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
    assert diff_type in ["log_odds", "diff"]
    assert rc_handling in ["average", "maximum"]
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

    preds = {}
    for k in seqs:
        # preds[k] = model.predict(seqs[k])
        preds[k] = np.array(model.predict(seqs[k])[..., output_filter_mask])

    if diff_type == "log_odds":
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
    elif diff_type == "diff":
        diffs = preds["alt"] - preds["ref"]
        diffs_rc = preds["alt_rc"] - preds["ref_rc"]

    if rc_handling == "average":
        diffs = np.mean([diffs, diffs_rc], axis=0)
    elif rc_handling == "maximum":
        replace_filt = np.abs(diffs) < np.abs(diffs_rc)
        diffs[replace_filt] = diffs_rc[replace_filt]

    diffs = pd.DataFrame(diffs, columns=out_annotation)

    return {"ism": diffs}
