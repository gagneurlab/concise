"""Helper functions for loading data from the attract db
"""
from concise.utils.pwm import PWM, load_motif_db
import numpy as np
import pandas as pd

from pkg_resources import resource_filename
HOCOMOCO_PWM = resource_filename('concise', 'resources/HOCOMOCOv10_pcms_HUMAN_mono.txt')


def get_metadata():
    """
    Get pandas.DataFrame with metadata about the PWM's. Columns:

    - PWM_id (id of the PWM - pass to get_pwm_list() for getting the pwm
    - TF
    - Organism
    - DB
    - Info
    - consensus
    """

    motifs = load_motif_db(HOCOMOCO_PWM)
    motif_names = sorted(list(motifs.keys()))

    df = pd.Series(motif_names).str.split(pat="_|\\.", expand=True)
    df.rename(columns={0: "TF", 1: "Organism", 2: "DB", 3: "info"}, inplace=True)

    # add PWM_id
    df.insert(0, "PWM_id", motif_names)

    # compute the consensus
    consensus = pd.Series([PWM(motifs[m]).get_consensus() for m in motif_names])
    df["consensus"] = consensus
    return df


def _normalize_pwm(pwm):
    rows = np.sum(pwm, axis=1)
    return pwm / rows.reshape([-1, 1])


def get_pwm_list(pwm_id_list, pseudocountProb=0.0001):
    """Get a list of HOCOMOCO PWM's.

    # Arguments
        pwm_id_list: List of id's from the `PWM_id` column in `get_metadata()` table
        pseudocountProb: Added pseudocount probabilities to the PWM

    # Returns
        List of `concise.utils.pwm.PWM` instances.
    """
    l = load_motif_db(HOCOMOCO_PWM)
    l = {k.split()[0]: v for k, v in l.items()}
    pwm_list = [PWM(_normalize_pwm(l[m]) + pseudocountProb, name=m) for m in pwm_id_list]
    return pwm_list
