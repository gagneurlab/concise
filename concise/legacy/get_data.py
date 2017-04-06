# functions for getting the data

import numpy as np
from concise.preprocessing import encodeDNA

# legacy
from concise.preprocessing.motifs import adjust_motifs


def prepare_data(dt, features, response, sequence, id_column=None, seq_align="end", trim_seq_len=None):
    """
    Prepare data for Concise.train or ConciseCV.train.

    Args:
        dt: A pandas DataFrame containing all the required data.
        features (List of strings): Column names of `dt` used to produce the features design matrix. These columns should be numeric.
        response (str or list of strings): Name(s) of column(s) used as a reponse variable.
        sequence (str): Name of the column storing the DNA/RNA sequences.
        id_column (str): Name of the column used as the row identifier.
        seq_align (str): one of ``{"start", "end"}``. To which end should we align sequences?
        trim_seq_len (int): Consider only first `trim_seq_len` bases of each sequence when generating the sequence design matrix. If :python:`None`, set :py:attr:`trim_seq_len` to the longest sequence length, hence whole sequences are considered.
        standardize_features (bool): If True, column in the returned matrix matrix :py:attr:`X_seq` are normalied to have zero mean and unit variance.


    Returns:
        tuple: Tuple with elements: :code:`(X_feat: X_seq, y, id_vec)`, where:

               - :py:attr:`X_feat`: features design matrix of shape :code:`(N, D)`, where N is :code:`len(dt)` and :code:`D = len(features)`
               - :py:attr:`X_seq`:  sequence matrix  of shape :code:`(N, 1, trim_seq_len, 4)`. It represents 1-hot encoding of the DNA/RNA sequence.
               - :py:attr:`y`: Response variable 1-column matrix of shape :code:`(N, 1)`    
               - :py:attr:`id_vec`: 1D Character array of shape :code:`(N)`. It represents the ID's of individual rows.

    Note:
        One-hot encoding  of the DNA/RNA sequence is the following:

        .. code:: python

               {
                 "A": np.array([1, 0, 0, 0]),
                 "C": np.array([0, 1, 0, 0]),
                 "G": np.array([0, 0, 1, 0]),
                 "T": np.array([0, 0, 0, 1]),
                 "U": np.array([0, 0, 0, 1]),
                 "N": np.array([0, 0, 0, 0]),
               }

    """
    if type(response) is str:
        response = [response]

    X_feat = np.array(dt[features], dtype="float32")
    y = np.array(dt[response], dtype="float32")
    X_seq = encodeDNA(seq_vec=dt[sequence],
                      maxlen=trim_seq_len,
                      seq_align=seq_align)
    X_seq = np.array(X_seq, dtype="float32")
    id_vec = np.array(dt[id_column])

    return X_feat, X_seq, y, id_vec
