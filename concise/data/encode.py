from concise.utils.pwm import PWM, load_motif_db
import pandas as pd

from pkg_resources import resource_filename
ENCODE_PWM = resource_filename('concise', 'resources/encode_motifs.txt.gz')

# TODO - move the stuff to concise.data.pwm


def _load_motifs():
    return load_motif_db(ENCODE_PWM, skipn_matrix=2)


def get_metadata():
    """Get pandas.DataFrame with metadata about the PWM's. Columns:

    - PWM_id (id of the PWM - pass to get_pwm_list() for getting the pwm
    - info1 - additional information about the motifs
    - info2
    - consensus: PWM consensus sequence
    """
    motifs = _load_motifs()

    motif_names = sorted(list(motifs.keys()))
    df = pd.Series(motif_names).str.split(expand=True)
    df.rename(columns={0: "PWM_id", 1: "info1", 2: "info2"}, inplace=True)

    # compute the consensus
    consensus = pd.Series([PWM(motifs[m]).get_consensus() for m in motif_names])
    df["consensus"] = consensus
    return df


def get_pwm_list(motif_name_list, pseudocountProb=0.0001):
    """Get a list of ENCODE PWM's.

    # Arguments
        pwm_id_list: List of id's from the `PWM_id` column in `get_metadata()` table
        pseudocountProb: Added pseudocount probabilities to the PWM

    # Returns
        List of `concise.utils.pwm.PWM` instances.
    """
    l = _load_motifs()
    l = {k.split()[0]: v for k, v in l.items()}
    pwm_list = [PWM(l[m] + pseudocountProb, name=m) for m in motif_name_list]
    return pwm_list
