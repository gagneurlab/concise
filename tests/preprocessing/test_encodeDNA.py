import pytest
from concise.preprocessing import encodeDNA
import numpy as np

def test_lengths():

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
