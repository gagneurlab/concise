import sklearn.preprocessing
import numpy as np

# vocabularies:
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
AMINO_ACIDS = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G", "H",
               "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
CODONS = ["AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA",
          "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT", "CAA", "CAC",
          "CAG", "CAT", "CCA", "CCC", "CCG", "CCT", "CGA", "CGC", "CGG",
          "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG", "GAT",
          "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA",
          "GTC", "GTG", "GTT", "TAC", "TAT", "TCA", "TCC", "TCG", "TCT",
          "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT"]
STOP_CODONS = ["TAG", "TAA", "TGA"]


def _get_vocab_dict(vocab):
    return {l: i for i, l in enumerate(vocab)}


def _get_index_dict(vocab):
    return {i: l for i, l in enumerate(vocab)}


def one_hot2token(arr):
    return arr.argmax(axis=2)


# TODO - take into account the neutral vocab
def one_hot2string(arr, vocab):
    """Convert a one-hot encoded array back to string
    """
    tokens = one_hot2token(arr)
    indexToLetter = _get_index_dict(vocab)

    return [''.join([indexToLetter[x] for x in row]) for row in tokens]


def tokenize(seq, vocab, neutral_vocab=[]):
    """Convert sequence to integers

    # Arguments
       seq: Sequence to encode
       vocab: Vocabulary to use
       neutral_vocab: Neutral vocabulary -> assign those values to -1

    # Returns
       List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    # Req: all vocabs have the same length
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]

    nchar = len(vocab[0])
    for l in vocab + neutral_vocab:
        assert len(l) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    vocab_dict = _get_vocab_dict(vocab)
    for l in neutral_vocab:
        vocab_dict[l] = -1

    # current performance bottleneck
    return [vocab_dict[seq[(i * nchar):((i + 1) * nchar)]] for i in range(len(seq) // nchar)]


# 512 ms vs 121 -> 4x slower than custom token2one_hot
# def token2one_hot(tvec, vocab_size):
#     """
#     Note: everything out of the vucabulary is transformed into `np.zeros(vocab_size)`
#     """
#     # This costs the most - memory allocation?
#     lb = sklearn.preprocessing.LabelBinarizer()
#     lb.fit(range(vocab_size))
#     return lb.transform(tvec)
#     # alternatively:
#     # return sklearn.preprocessing.label_binarize(tvec, list(range(vocab_size)))


def token2one_hot(tvec, vocab_size):
    """
    Note: everything out of the vucabulary is transformed into `np.zeros(vocab_size)`
    """
    arr = np.zeros((len(tvec), vocab_size))

    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec_range[tvec >= 0], tvec[tvec >= 0]] = 1
    return arr


def encodeSequence(seq_vec, vocab, neutral_vocab, maxlen=None,
                   seq_align="start", pad_value="N", encode_type="one_hot"):
    """Convert a list of genetic sequences into one-hot-encoded array.

    # Arguments
       seq_vec: list of strings (genetic sequences)
       vocab: list of chars: List of "words" to use as the vocabulary. Can be strings of length>0,
            but all need to have the same length. For DNA, this is: ["A", "C", "G", "T"].
       neutral_vocab: list of chars: Values used to pad the sequence or represent unknown-values. For DNA, this is: ["N"].
       maxlen: int or None,
            Should we trim (subset) the resulting sequence. If None don't trim.
            Note that trims wrt the align parameter.
            It should be smaller than the longest sequence.
       seq_align: character; 'end' or 'start'
            To which end should we align sequences?
       encode_type: "one_hot" or "token". "token" represents each vocab element as a positive integer from 1 to len(vocab) + 1.
                  neutral_vocab is represented with 0.

    # Returns
        Array with shape for encode_type:

            - "one_hot": `(len(seq_vec), maxlen, len(vocab))`
            - "token": `(len(seq_vec), maxlen)`

        If `maxlen=None`, it gets the value of the longest sequence length from `seq_vec`.
    """
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seq_vec, str):
        raise ValueError("seq_vec should be an iterable returning " +
                         "strings not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab

    assert encode_type in ["one_hot", "token"]

    seq_vec = pad_sequences(seq_vec, maxlen=maxlen,
                            align=seq_align, value=pad_value)

    if encode_type == "one_hot":
        arr_list = [token2one_hot(tokenize(seq, vocab, neutral_vocab), len(vocab))
                    for i, seq in enumerate(seq_vec)]
    elif encode_type == "token":
        arr_list = [1 + np.array(tokenize(seq, vocab, neutral_vocab)) for seq in seq_vec]
        # we add 1 to be compatible with keras: https://keras.io/layers/embeddings/
        # indexes > 0, 0 = padding element

    return np.stack(arr_list)


def encodeDNA(seq_vec, maxlen=None, seq_align="start"):
    """Convert the DNA sequence into 1-hot-encoding numpy array

    # Arguments
        seq_vec: list of chars
            List of sequences that can have different lengths

        maxlen: int or None,
            Should we trim (subset) the resulting sequence. If None don't trim.
            Note that trims wrt the align parameter.
            It should be smaller than the longest sequence.

        seq_align: character; 'end' or 'start'
            To which end should we align sequences?

    # Returns
        3D numpy array of shape (len(seq_vec), trim_seq_len(or maximal sequence length if None), 4)

    # Example

        ```python
            >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
            >>> X_seq = encodeDNA(sequence_vec, seq_align="end", maxlen=8)
            >>> X_seq.shape
            (2, 8, 4)

            >>> print(X_seq)
            [[[0 0 0 1]
              [1 0 0 0]
              [0 1 0 0]
              [0 0 0 1]
              [0 1 0 0]
              [1 0 0 0]
              [0 0 1 0]
              [1 0 0 0]]

             [[0 0 0 0]
              [0 0 0 0]
              [0 0 0 1]
              [0 1 0 0]
              [0 0 0 1]
              [0 0 0 1]
              [0 0 0 1]
              [1 0 0 0]]]
        ```
    """
    return encodeSequence(seq_vec,
                          vocab=DNA,
                          neutral_vocab="N",
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="N",
                          encode_type="one_hot")


def encodeRNA(seq_vec, maxlen=None, seq_align="start"):
    """Convert the RNA sequence into 1-hot-encoding numpy array as for encodeDNA
    """
    return encodeSequence(seq_vec,
                          vocab=RNA,
                          neutral_vocab="N",
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="N",
                          encode_type="one_hot")


def encodeCodon(seq_vec, ignore_stop_codons=True, maxlen=None, seq_align="start", encode_type="one_hot"):
    """Convert the Codon sequence into 1-hot-encoding numpy array

    # Arguments
        seq_vec: List of strings/DNA sequences
        ignore_stop_codons: boolean; if True, STOP_CODONS are omitted from one-hot encoding.
        maxlen: Maximum sequence length. See `pad_sequences` for more detail
        seq_align: How to align the sequences of variable lengths. See `pad_sequences` for more detail
        encode_type: can be `"one_hot"` or `token` for token encoding of codons (incremental integer ).

    # Returns
        numpy.ndarray of shape `(len(seq_vec), maxlen / 3, 61 if ignore_stop_codons else 64)`
    """
    if ignore_stop_codons:
        vocab = CODONS
        neutral_vocab = STOP_CODONS + ["NNN"]
    else:
        vocab = CODONS + STOP_CODONS
        neutral_vocab = ["NNN"]

    # replace all U's with A's?
    seq_vec = [str(seq).replace("U", "T") for seq in seq_vec]

    return encodeSequence(seq_vec,
                          vocab=vocab,
                          neutral_vocab=neutral_vocab,
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="NNN",
                          encode_type=encode_type)


def encodeAA(seq_vec, maxlen=None, seq_align="start", encode_type="one_hot"):
    """Convert the Amino-acid sequence into 1-hot-encoding numpy array

    # Arguments
        seq_vec: List of strings/amino-acid sequences
        maxlen: Maximum sequence length. See `pad_sequences` for more detail
        seq_align: How to align the sequences of variable lengths. See `pad_sequences` for more detail
        encode_type: can be `"one_hot"` or `token` for token encoding of codons (incremental integer ).

    # Returns
        numpy.ndarray of shape `(len(seq_vec), maxlen, 22)`
    """
    return encodeSequence(seq_vec,
                          vocab=AMINO_ACIDS,
                          neutral_vocab="_",
                          maxlen=maxlen,
                          seq_align=seq_align,
                          pad_value="_",
                          encode_type=encode_type)


def pad_sequences(sequence_vec, maxlen=None, align="end", value="N"):
    """Pad and/or trim a list of sequences to have common length. Procedure:

        1. Pad the sequence with N's or any other string or list element (`value`)
        2. Subset the sequence

    # Note
        See also: https://keras.io/preprocessing/sequence/
        Aplicable also for lists of characters

    # Arguments
        sequence_vec: list of chars or lists
            List of sequences that can have various lengths
        value: Neutral element to pad the sequence with. Can be `str` or `list`.
        maxlen: int or None; Final lenght of sequences.
             If None, maxlen is set to the longest sequence length.
        align: character; 'start', 'end' or 'center'
            To which end to align the sequences when triming/padding. See examples bellow.

    # Returns
        List of sequences of the same class as sequence_vec

    # Example

        ```python
            >>> sequence_vec = ['CTTACTCAGA', 'TCTTTA']
            >>> pad_sequences(sequence_vec, 10, align="start", value="N")
            ['CTTACTCAGA', 'TCTTTANNNN']
            >>> pad_sequences(sequence_vec, 10, align="end", value="N")
            ['CTTACTCAGA', 'NNNNTCTTTA']
            >>> pad_sequences(sequence_vec, 4, align="center", value="N")
            ['ACTC', 'CTTT']
        ```
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(sequence_vec[0]))
    assert not isinstance(sequence_vec, str)
    assert isinstance(sequence_vec[0], list) or isinstance(sequence_vec[0], str)

    max_seq_len = max([len(seq) for seq in sequence_vec])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        print("WARNING: Maximum sequence length (%s) is less than maxlen (%s)" % (max_seq_len, maxlen))
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in sequence_vec:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    # pad and subset
    def pad(seq, max_seq_len, value="N", align="end"):
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

        # normalize for the length
        n_left = n_left // len(value)
        n_right = n_right // len(value)

        return value * n_left + seq + value * n_right

    def trim(seq, maxlen, align="end"):
        seq_len = len(seq)

        assert maxlen <= seq_len
        if align is "end":
            return seq[-maxlen:]
        elif align is "start":
            return seq[0:maxlen]
        elif align is "center":
            dl = seq_len - maxlen
            n_left = dl // 2 + dl % 2
            n_right = seq_len - dl // 2
            return seq[n_left:n_right]
        else:
            raise ValueError("align can be of: end, start or center")

    padded_sequence_vec = [pad(seq, max(max_seq_len, maxlen),
                               value=value, align=align) for seq in sequence_vec]
    padded_sequence_vec = [trim(seq, maxlen, align=align) for seq in padded_sequence_vec]

    return padded_sequence_vec
