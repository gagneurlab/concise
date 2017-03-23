import numpy as np
from .dna import seq2numpy


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
