import numpy as np


# TODO - make use_keras a default choice, adopt concise


def encodeDNA(seq_vec, trim_seq_len=None, seq_align="start", use_keras=False):
    """
    Convert the DNA sequence to 1-hot-encoding numpy array

    parameters
    ----------
    seq_vec: list of chars
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
    x = seq2numpy(seq_vec)

    if use_keras:
        x = np.squeeze(x, 1)
    return x


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
