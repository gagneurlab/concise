"""Helper functions for loading data from the attract db
"""
from concise.utils.pwm import PWM, load_motif_db
import pandas as pd

from pkg_resources import resource_filename
ATTRACT_METADTA = resource_filename('concise', 'resources/attract_metadata.txt')
ATTRACT_PWM = resource_filename('concise', 'resources/attract_pwm.txt')


def get_metadata():
    """
    Get pandas.DataFrame with metadata about the Attract PWM's. Columns:

    - PWM_id (id of the PWM - pass to get_pwm_list() for getting the pwm
    - Gene_name
    - Gene_id
    - Mutated	(if the target gene is mutated)
    - Organism
    - Motif     (concsensus motif)
    - Len	(lenght of the motif)
    - Experiment_description(when available)
    - Database (Database from where the motifs were extracted PDB: Protein data bank, C: Cisbp-RNA, R:RBPDB, S: Spliceaid-F, AEDB:ASD)
    - Pubmed (pubmed ID)
    - Experiment (type of experiment; short description)
    - Family (domain)
    - Score (Qscore refer to the paper)
    """
    dt = pd.read_table(ATTRACT_METADTA)
    dt.rename(columns={"Matrix_id": "PWM_id"}, inplace=True)
    # put to firt place
    cols = ['PWM_id'] + [col for col in dt if col != 'PWM_id']

    # rename Matrix_id to PWM_id
    return dt[cols]


def get_pwm_list(pwm_id_list, pseudocountProb=0.0001):
    """Get a list of Attract PWM's.

    # Arguments
        pwm_id_list: List of id's from the `PWM_id` column in `get_metadata()` table
        pseudocountProb: Added pseudocount probabilities to the PWM

    # Returns
        List of `concise.utils.pwm.PWM` instances.
    """
    l = load_motif_db(ATTRACT_PWM)
    l = {k.split()[0]: v for k, v in l.items()}
    pwm_list = [PWM(l[str(m)] + pseudocountProb, name=m) for m in pwm_id_list]
    return pwm_list
