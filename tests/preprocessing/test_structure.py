#
import numpy as np
from concise.preprocessing import encodeRNAStructure
from concise.preprocessing.structure import run_RNAplfold, read_RNAplfold


def test_data():
    input_fasta = "tests/preprocessing/RNAplfold_data/input.fasta"
    tmpdir = "/tmp/RNAplfold"

    run_RNAplfold(input_fasta, tmpdir)

    arr = read_RNAplfold(tmpdir)

    # all values sum to 1
    assert np.allclose(arr.sum(axis=2), 1)
    assert arr.shape == (2, 302, 5)
    # shape = chanells, sequences, values
    #
    # what we want: seqs, values, chanells?
    seq = ["TATTATGTATATGTATA", "TATGTATAT"]

    arr = encodeRNAStructure(seq)
    assert np.allclose(arr.sum(axis=2), 1)
    assert arr.shape == (2, 17, 5)
