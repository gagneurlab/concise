"""Helper functions for loading data from the attract db
"""
# from simdna.synthetic.loadedmotifs import AbstractLoadedMotifsFromFile
from simdna.synthetic import AbstractLoadedMotifsFromFile
from simdna import pwm
from simdna import util
from concise.utils.pwm import PWM
import pandas as pd

from pkg_resources import resource_filename
ATTRACT_METADTA = resource_filename('concise', 'resources/attract_metadata.txt')
ATTRACT_PWM = resource_filename('concise', 'resources/attract_pwm.txt')


class LoadedAttractMotifs(AbstractLoadedMotifsFromFile):
    """A class for reading in a motifs file in the ENCODE motifs format.
    This class is specifically for reading files in the encode motif
    format - specifically the motifs.txt file that contains Pouya's motifs
    (http://compbio.mit.edu/encode-motifs/motifs.txt)
    Basically, the motif declarations start with a >, the first
    characters after > until the first space are taken as the motif name,
    the lines after the line with a > have the format:
    "<ignored character> <prob of A> <prob of C> <prob of G> <prob of T>"
    """

    def getReadPwmAction(self, loadedMotifs):
        """See superclass.
        """
        currentPwm = util.VariableWrapper(None)

        def action(inp, lineNumber):
            if (inp.startswith(">")):
                inp = inp.lstrip(">")
                inpArr = inp.split()
                motifName = inpArr[0]
                currentPwm.var = pwm.PWM(motifName)
                loadedMotifs[currentPwm.var.name] = currentPwm.var
            else:
                # assume that it's a line of the pwm
                assert currentPwm.var is not None
                inpArr = inp.split()
                row = [float(x) for x in inpArr]
                rowNormalized = [e / sum(row) for e in row]
                # there is a small typo in 4 rows. fix them by hand
                # if not abs(sum(row) - 1.0) < 0.0001:
                #     print(row)
                # [0.39, 0.2, 0.01, 0.39]
                # [0.01, 0.2, 0.39, 0.39]
                # [0.418571428571, 0.418571428571, 0.142857142857, 0.01]
                # [0.01, 0.48, 0.01, 0.48]
                # [0.378888888889, 0.222222222222, 0.378888888889, 0.01]
                currentPwm.var.addRow(rowNormalized)
        return action


def get_metadata():
    """
    Get pandas.DataFrame with metadata about the PWM's.

    Columns:
        PWM_id (id of the PWM - pass to get_pwm_list() for getting the pwm
        Gene_name
        Gene_id
        Mutated	(if the target gene is mutated)
        Organism
        Motif     (concsensus motif)
        Len	(lenght of the motif)
        Experiment_description(when available)
        Database (Database from where the motifs were extracted PDB: Protein data bank, C: Cisbp-RNA, R:RBPDB, S: Spliceaid-F, AEDB:ASD)
        Pubmed (pubmed ID)
        Experiment (type of experiment; short description)
        Family (domain)
        Score (Qscore refer to the paper)
    """
    dt = pd.read_table(ATTRACT_METADTA)
    dt.rename(columns={"Matrix_id": "PWM_id"}, inplace=True)
    # put to firt place
    cols = ['PWM_id'] + [col for col in dt if col != 'PWM_id']

    # rename Matrix_id to PWM_id
    return dt[cols]


def get_pwm_list(pwm_id_list, pseudocountProb=0.0001):
    l = LoadedAttractMotifs(ATTRACT_PWM, pseudocountProb=pseudocountProb)

    # sidna_pwm_list = [l[m_id] for m_id in matrix_id_list]
    pwm_list = [PWM(l.loadedMotifs[str(m_id)].getRows(), name=m_id) for m_id in pwm_id_list]
    return pwm_list
