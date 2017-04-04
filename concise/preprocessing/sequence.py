import numpy as np


def encodeDNA(seq_vec, trim_seq_len=None, seq_align="start"):
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
    if isinstance(seq_vec, str):
        raise ValueError("seq_vec should be an iterable returning " +
                         "strings not a string itself")
    seq_vec = pad_and_trim(seq_vec, neutral_element="N",
                           align=seq_align, target_seq_len=trim_seq_len)

    x = seq2numpy(seq_vec)
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
        seq_tensor.append(dna_seq_to_1hot(seq))

    return np.stack(seq_tensor)


def pad_and_trim(sequence_vec, neutral_element="N", target_seq_len=None, align="end"):
    """
    1. Pad the sequence with N's or any other sequence element
    2. Subset the sequence

    parameters
    ----------
    sequence_vec: list of chars
        List of sequences that can have different lengths
    neutral_element:
        Neutral element to pad the sequence with
    target_seq_len: int or None,
        Should we trim (subset) the resulting sequence. If None don't trim.
        Note that trims wrt the align parameter.
        It should be smaller than the longest sequence.
    align: character; 'end' or 'start'
        To which end should we align sequences?

    returns
    -------
    List of sequences of the same class as sequence_vec

    Examples
    --------
    >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
    >>> seq_pad_and_trim(sequence_vec, "N", 10, "start")
    """

    # neutral element type checkoing
    assert len(neutral_element) == 1
    assert isinstance(neutral_element, type(sequence_vec[0]))
    assert isinstance(neutral_element, list) or isinstance(neutral_element, str)
    assert not isinstance(sequence_vec, str)
    assert isinstance(sequence_vec[0], list) or isinstance(sequence_vec[0], str)

    max_seq_len = max([len(seq) for seq in sequence_vec])

    if target_seq_len is None:
        target_seq_len = max_seq_len
    else:
        target_seq_len = int(target_seq_len)

    if max_seq_len < target_seq_len:
        print("WARNING: Maximum sequence length (%s) is less than target_seq_len (%s)" % (max_seq_len, target_seq_len))
        max_seq_len = target_seq_len

        # pad and subset

    def pad(seq, max_seq_len, neutral_element="N", align="end"):
        seq_len = len(seq)
        assert max_seq_len >= seq_len
        if align is "end":
            n_left = max_seq_len - seq_len
            n_right = 0
        elif align is "start":
            n_right = max_seq_len - seq_len
            n_left = 0
        elif align is "center":
            n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
            n_right = (max_seq_len - seq_len) // 2
        else:
            raise ValueError("align can be of: end, start or center")
        return neutral_element * n_left + seq + neutral_element * n_right

    def trim(seq, target_seq_len, align="end"):
        seq_len = len(seq)

        assert target_seq_len <= seq_len
        if align is "end":
            return seq[-target_seq_len:]
        elif align is "start":
            return seq[0:target_seq_len]
        elif align is "center":
            dl = seq_len - target_seq_len
            n_left = dl // 2 + dl % 2
            n_right = seq_len - dl // 2
            return seq[n_left:-n_right]
        else:
            raise ValueError("align can be of: end, start or center")

    padded_sequence_vec = [pad(seq, max(max_seq_len, target_seq_len),
                               neutral_element=neutral_element, align=align) for seq in sequence_vec]
    padded_sequence_vec = [trim(seq, target_seq_len, align=align) for seq in padded_sequence_vec]

    return padded_sequence_vec
