import os
import numpy as np
from concise.preprocessing import encodeRNAStructure
from concise.preprocessing.structure import run_RNAplfold, read_RNAplfold, encodeRNAStructure_parallel
import pandas as pd
from contextlib import contextmanager


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def manual_test_basic():
    input_fasta = "tests/preprocessing/RNAplfold_data/input.fasta"
    tmpdir = "/tmp/RNAplfold"

    run_RNAplfold(input_fasta, tmpdir)

    arr = read_RNAplfold(tmpdir)

    # all values sum to 1
    assert np.allclose(arr.sum(axis=2), 1)
    assert arr.shape == (2, 302, 5)
    # shape = chanells, sequences, values


def test_encodeRNAstructure():
    with cd("/tmp/"):
        # what we want: seqs, values, chanells?
        seq = ["TATTATGTATATGTATA", "TATGTATAT"]

        arr = encodeRNAStructure(seq)
    assert np.allclose(arr.sum(axis=2), 1)
    assert arr.shape == (2, 17, 5)


def test_other_objects():
    seq = np.array(["TATTATGTATATGTATA", "TATGTATAT"])

    arr = encodeRNAStructure(seq)
    assert np.allclose(arr.sum(axis=2), 1)
    assert arr.shape == (2, 17, 5)


def test_real_data():
    csv_file_path = "data/pombe_half-life_UTR3.csv"
    dt = pd.read_csv(csv_file_path)
    seq_vec = dt["seq"][:6]
    a = encodeRNAStructure(seq_vec)
    assert a.shape == (6, seq_vec.str.len().max(), 5)


# def test_real_data_parallel():
#     csv_file_path = "data/pombe_half-life_UTR3.csv"
#     dt = pd.read_csv(csv_file_path)
#     seq_vec = dt["seq"][:6]
#     a = encodeRNAStructure_parallel(seq_vec, n_cores=4)
#     assert a.shape == (6, seq_vec.str.len().max(), 5)


