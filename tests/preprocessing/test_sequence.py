import pytest
from concise.preprocessing import encodeDNA
from concise.preprocessing.sequence import pad_and_trim
import numpy as np


def test_encodeDNA():

    seq = "ACGTTTATNU"
    assert len(seq) == 10

    with pytest.raises(ValueError):
        encodeDNA(seq)

    assert encodeDNA([seq]).shape == (1, 10, 4)

    assert encodeDNA([seq], trim_seq_len=20).shape == (1, 20, 4)

    assert encodeDNA([seq], trim_seq_len=5).shape == (1, 5, 4)
    assert np.all(encodeDNA([seq])[0, 0] == np.array([1, 0, 0, 0]))
    assert np.all(encodeDNA([seq])[0, 1] == np.array([0, 1, 0, 0]))
    assert np.all(encodeDNA([seq])[0, 2] == np.array([0, 0, 1, 0]))
    assert np.all(encodeDNA([seq])[0, 3] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, 4] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, -1] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, -2] == np.array([0, 0, 0, 0]))


def test_pad_and_trim():
    sequence_vec = ["ACGTTTATNU"]
    assert len(pad_and_trim(sequence_vec, neutral_element="N",
                            target_seq_len=20, align="end")[0]) is 20

    # works with lists
    assert pad_and_trim([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], neutral_element=[0],
                        target_seq_len=5) == [[0, 0, 1, 2, 3],
                                              [0, 2, 2, 3, 4],
                                              [0, 0, 0, 31, 3],
                                              [0, 0, 0, 4, 2]]

    assert pad_and_trim([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], neutral_element=[0],
                        target_seq_len=2, align="end") == [[2, 3],
                                                           [3, 4],
                                                           [31, 3],
                                                           [4, 2]]

    assert pad_and_trim([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], neutral_element=[0],
                        target_seq_len=2, align="start") == [[1, 2],
                                                             [2, 2],
                                                             [31, 3],
                                                             [4, 2]]
