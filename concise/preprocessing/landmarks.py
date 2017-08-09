import numpy as np
import pandas as pd
from copy import deepcopy

# TODO - just use pandas to do so
def pd_read_gtf(path):
    gtf = pd.read_table(path,
                        header=None,
                        comment="#",
                        index_col=False,
                        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame"],
                        skip_blank_lines=True)
    return gtf


gtf = pd_read_gtf(gtf_path)


def range_start(gtf):
    pos = np.where(gtf.strand == "-", gtf.end, gtf.start)
    return pd.DataFrame.from_items([("seqname", gtf.seqname),
                                    ("pos", pos),
                                    ("strand", gtf.strand)])

def range_end(gtf):
    pos = np.where(gtf.strand == "-", gtf.start, gtf.end)
    return pd.DataFrame.from_items([("seqname", gtf.seqname),
                                    ("pos", pos),
                                    ("strand", gtf.strand)])


def landmark_tss(gr):
    return range_start(gr[gr.feature == "transcript"])


def landmark_polya(gr):
    return range_end(gr[gr.feature == "transcript"])


def landmark_exon_intron(gr):
    # TODO filter transforitps
    # remove exons that are also on the transcript ends
    # gr_exon_intron = gr_exon_intron[!gr_exon_intron % in % gr_polya]
    # gr_intron_exon = gr_intron_exon[!gr_intron_exon % in % gr_TSS]
    return range_end(gr[gr.feature == "exon"])


def landmark_intron_exon(gr):
    # TODO - filter transripts
    # remove exons that are also on the transcript ends
    # gr_exon_intron = gr_exon_intron[!gr_exon_intron % in % gr_polya]
    # gr_intron_exon = gr_intron_exon[!gr_intron_exon % in % gr_TSS]
    return range_start(gr[gr.feature == "exon"])


def landmark_start_codon(gr):
    # TODO - filter transripts
    return range_start(gr[gr.feature == "start_codon"])


def landmark_stop_codon(gr):
    # TODO - filter transripts
    return range_end(gr[gr.feature == "stop_codon"])


def landmark_gene_start(gr):
    return range_start(gr[gr.feature == "gene"])


def landmark_gene_end(gr):
    return range_end(gr[gr.feature == "gene"])


def tss(gtffile):
    # Code from HTSeq
    tsspos = set()
    for feature in gtffile:
        if feature.type == "exon" and feature.attr["exon_number"] == "1":
            tsspos.add(feature.iv.start_d_as_pos)
    return tsspos
