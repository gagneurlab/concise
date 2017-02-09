# functions for getting the data

import numpy as np
# function definitions

def dna_seq_to_1hot(seq):
    """
    Convert the DNA string into 1-hot encoded numpy array

    Inputs:
    - seq string
    """
    hash_dict = {"A": np.array([1, 0, 0, 0]),
                 "C": np.array([0, 1, 0, 0]),
                 "G": np.array([0, 0, 1, 0]),
                 "T": np.array([0, 0, 0, 1]),
                 "U": np.array([0, 0, 0, 1]),  # RNA sequence support
                 "N": np.array([0, 0, 0, 0])}

    DNA_list = []
    for char in seq:
        DNA_list.append(hash_dict[char])

    final = np.asarray(DNA_list).astype(np.float32)
    return final


def seq2numpy(sequence_vec):
    """
    Convert a list of sequences into a numpy 4D tensor
    """

    seq_tensor = []
    for seq in sequence_vec:
        seq_tensor.append(dna_seq_to_1hot(seq)[np.newaxis])

    return np.stack(seq_tensor)


def dna_seq_array(seq_vec, seq_align="end", trim_seq_len=None):
    """
    Convert the DNA sequence to 1-hot-encoding numpy array

    parameters
    ----------
    sequence_vec: list of chars
        List of sequences that can have different lengths

    seq_align: character; 'end' or 'start'
        To which end should we align sequences?

    trim_seq_len: int or None, 
        Should we trim (subset) the resulting sequence. If None don't trim.
        Note that trims wrt the align parameter.
        It should be smaller than the longest sequence.

    returns
    -------
    4D numpy array of shape (len(seq_vec), 1, trim_seq_len(or maximal sequence length if None), 4)

    Examples
    --------
    >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
    >>> X_seq = dna_seq_list_to_1hot(sequence_vec, "end", trim_seq_len = 8)
    >>> X_seq.shape
    (2, 1, 8, 4)

    >>> print(X_seq)
    [[[[ 0.  0.  0.  1.]
       [ 1.  0.  0.  0.]
       [ 0.  1.  0.  0.]
       [ 0.  0.  0.  1.]
       [ 0.  1.  0.  0.]
       [ 1.  0.  0.  0.]
       [ 0.  0.  1.  0.]
       [ 1.  0.  0.  0.]]]


     [[[ 0.  0.  0.  0.]
       [ 0.  0.  0.  0.]
       [ 0.  0.  0.  1.]
       [ 0.  1.  0.  0.]
       [ 0.  0.  0.  1.]
       [ 0.  0.  0.  1.]
       [ 0.  0.  0.  1.]
       [ 1.  0.  0.  0.]]]]
    """

    seq_vec = seq_pad_and_trim(seq_vec, seq_align=seq_align, trim_seq_len=trim_seq_len)
    return seq2numpy(seq_vec)


#####
# used function
#####
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
    X_seq = dna_seq_array(seq_vec=dt[sequence],
                          seq_align=seq_align,
                          trim_seq_len=trim_seq_len)
    X_seq = np.array(X_seq, dtype="float32")
    id_vec = np.array(dt[id_column])

    return X_feat, X_seq, y, id_vec


def seq_pad_and_trim(sequence_vec, seq_align="end", trim_seq_len=None):
    """
    1. Pad the sequence with N's
    2. Subset the sequence

    parameters
    ----------
    sequence_vec: list of chars
        List of sequences that can have different lengths

    seq_align: character; 'end' or 'start'
        To which end should we align sequences?

    trim_seq_len: int or None, 
        Should we trim (subset) the resulting sequence. If None don't trim.
        Note that trims wrt the align parameter.
        It should be smaller than the longest sequence.

    returns
    -------
    List of sequences

    Examples
    --------
    >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
    >>> seq_pad_and_trim(sequence_vec, "start", 8)
    """

    max_seq_len = max([len(seq) for seq in sequence_vec])

    if trim_seq_len is None:
        trim_seq_len = max_seq_len
    else:
        trim_seq_len = int(trim_seq_len)

    if max_seq_len < trim_seq_len:
        raise Warning("Maximum sequence length (%s) is less than trim_seq_len (%s)" % (max_seq_len, trim_seq_len))

        # pad and subset
    if seq_align == "end":
        # pad
        padded_sequence_vec = [seq.rjust(max_seq_len, str("N")) for seq in sequence_vec]
        # trim
        padded_sequence_vec = [seq[-trim_seq_len:] for seq in padded_sequence_vec]
    elif seq_align == "start":
        # pad
        padded_sequence_vec = [seq.ljust(max_seq_len, str("N")) for seq in sequence_vec]
        # trim
        padded_sequence_vec = [seq[0:trim_seq_len] for seq in padded_sequence_vec]
    else:
        raise TypeError("seq_align can only be 'start' or 'end'")

    return padded_sequence_vec

# --------------------------------------------


def adjust_motifs(motifs, filter_width, n_motifs):

    # helper function
    def adjust_padding_single_motif(motif, filter_width):
        cur_filter_width = len(motif)

        # subsample the motif
        if cur_filter_width > filter_width:
            motif = motif[:filter_width]

        # extend the motif with N's
        if cur_filter_width < filter_width:
            missing = filter_width - cur_filter_width
            n_pad_start = missing // 2
            n_pad_end = missing // 2 + missing % 2

            motif = 'N' * n_pad_start + motif + 'N' * n_pad_end

        return motif

    # padd the motifs
    motifs = [adjust_padding_single_motif(motif, filter_width) for motif in motifs]

    cur_n_motifs = len(motifs)
    # extend or cut the motifs
    if cur_n_motifs > n_motifs:
        motifs = motifs[:n_motifs]

    if cur_n_motifs < n_motifs:
        new_motifs = ['N' * filter_width for i in range(n_motifs - cur_n_motifs)]
        motifs = motifs + new_motifs

    return motifs

def intial_motif_filter(motifs):
    filter_initial = np.rollaxis(seq2numpy(motifs), 0, 4)
    return filter_initial

# swap some axes

def convert_motif_arrays(filter_initial):
    return np.swapaxes(filter_initial[0], 0, 2)

