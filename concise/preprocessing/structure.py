"""Scripts for secondary structure
"""

# read in the output of the structural context

import numpy as np
from concise.utils.fasta import write_fasta, iter_fasta
from concise.preprocessing.sequence import pad_and_trim
from subprocess import call, run
import os

RNAplfold_BIN_DIR = "concise/resources/RNAplfold"
RNAplfold_PROFILES_EXECUTE = ["H", "I", "M", "E"]
RNAplfold_PROFILES = ["Pairedness", "Hairpin loop", "Internal loop", "Multi loop", "External region"]


# TODO detailed description of the parameters?
# TODO - generate uuid
def run_RNAplfold(input_fasta, tmpdir, W=240, L=160, U=1):
    """
    Arguments:
       W, Int: span - window length
       L, Int, maxiumm span
       U, Int, size of unpaired region

    Recomendation:
    - for human, mouse use W, L, u : 240, 160, 1
    - for fly, yeast   use W, L, u :  80,  40, 1

    """

    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    profiles = RNAplfold_PROFILES_EXECUTE
    for i, P in enumerate(profiles):
        print("running {P}_RNAplfold... ({i}/{N})".format(P=P, i=i + 1, N=len(profiles)))

        command = "{bin}/{P}_RNAplfold".format(bin=RNAplfold_BIN_DIR, P=P)
        args = " -W {W} -L {L} -u {U} < {fa} > {tmp}/{P}_profile.fa".format(P=P, W=W, L=L, U=U, fa=input_fasta, tmp=tmpdir)
        os.system(command + args)
    print("done!")


def read_RNAplfold(tmpdir, trim_seq_len=None, seq_align="start", pad_with=0.2):
    def read_profile(tmpdir, P):
        return [values.strip().split("\t")
                for seq_name, values in iter_fasta("{tmp}/{P}_profile.fa".format(tmp=tmpdir, P=P))]

    arr_hime = np.array([pad_and_trim(read_profile(tmpdir, P),
                                      neutral_element=[pad_with],
                                      align=seq_align,
                                      target_seq_len=trim_seq_len)
                         for P in RNAplfold_PROFILES_EXECUTE], dtype="float32")

    # add the pairness column
    arr_p = 1 - arr_hime.sum(axis=0)[np.newaxis]
    arr = np.concatenate((arr_p, arr_hime))

    # reshape to: seq, seq_length, num_channels
    arr = np.moveaxis(arr, 0, 2)
    return arr


def encodeRNAStructure(seq_vec, trim_seq_len=None, seq_align="start",
                       W=240, L=160, U=1,
                       tmpdir="/tmp/RNAplfold/"):
    """
    Arguments:
       W, Int: span - window length
       L, Int, maxiumm span
       U, Int, size of unpaired region

    Recomendation:
    - for human, mouse use W, L, u : 240, 160, 1
    - for fly, yeast   use W, L, u :  80,  40, 1

    """
    fasta_path = tmpdir + "/input.fasta"
    write_fasta(fasta_path, seq_vec)
    run_RNAplfold(fasta_path, tmpdir, W=W, L=L, U=U)
    return read_RNAplfold(tmpdir, trim_seq_len, seq_align, pad_with=.2)
