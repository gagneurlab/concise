import numpy as np
from .sequence import encodeDNA, pad_sequences


def adjust_motifs(motifs, filter_width, n_motifs):
    motifs = pad_sequences(motifs, maxlen=filter_width, align="center", value="N")

    cur_n_motifs = len(motifs)
    # extend or cut the motifs
    if cur_n_motifs > n_motifs:
        motifs = motifs[:n_motifs]

    if cur_n_motifs < n_motifs:
        new_motifs = ['N' * filter_width for i in range(n_motifs - cur_n_motifs)]
        motifs = motifs + new_motifs

    return motifs


def intial_motif_filter(motifs):
    filter_initial = np.rollaxis(encodeDNA(motifs), 0, 4)
    return filter_initial


def convert_motif_arrays(filter_initial):
    """swap some axes
    """
    return np.swapaxes(filter_initial[0], 0, 2)
