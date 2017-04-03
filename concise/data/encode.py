# 0.4
# from simdna.synthetic.loadedmotifs import LoadedEncodeMotifs
from simdna.synthetic import LoadedEncodeMotifs
from simdna import ENCODE_MOTIFS_PATH
from concise.utils.pwm import PWM
import pandas as pd


def get_metadata():
    loadedMotifs = LoadedEncodeMotifs(ENCODE_MOTIFS_PATH, pseudocountProb=0.001)

    motifs = sorted(list(loadedMotifs.loadedMotifs.keys()))
    consensus = [loadedMotifs.loadedMotifs[motif].bestPwmHit for motif in motifs]
    dt = pd.DataFrame({"motif_name": motifs, "consensus": consensus},
                      columns=["motif_name", "consensus"]
                      )

    return dt


def get_pwm_list(motif_name_list, pseudocountProb=0.0001):
    l = LoadedEncodeMotifs(ENCODE_MOTIFS_PATH, pseudocountProb=pseudocountProb)

    # sidna_pwm_list = [l[m_id] for m_id in matrix_id_list]
    pwm_list = [PWM(l.loadedMotifs[m_id].getRows(), name=m_id) for m_id in motif_name_list]
    return pwm_list
