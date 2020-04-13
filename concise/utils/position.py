import numpy as np
import pandas as pd
from copy import deepcopy
from concise.utils.helper import get_from_module, _to_string
import logging
from gtfparse import read_gtf

_logger = logging.getLogger('genomelake')

ALL_LANDMARKS = ["tss", "polya",
                 "exon_intron", "intron_exon",
                 "start_codon", "stop_codon",
                 "gene_start", "gene_end"]


def extract_landmarks(gtf, landmarks=ALL_LANDMARKS):
    """Given an gene annotation GFF/GTF file,

    # Arguments
        gtf: File path or a loaded `pd.DataFrame` with columns:
    seqname, feature, start, end, strand
        landmarks: list or a dictionary of landmark extractors (function or name)

    # Note
        When landmark extractor names are used, they have to be implemented in
    the module `concise.preprocessing.position`

    # Returns
        Dictionary of pd.DataFrames with landmark positions
    (columns: seqname, position, strand)
    """
    if isinstance(gtf, str):
        _logger.info("Reading gtf file..")
        gtf = read_gtf(gtf)
        _logger.info("Done")

    _logger.info("Running landmark extractors..")
    # landmarks to a dictionary with a function
    assert isinstance(landmarks, (list, tuple, set, dict))
    if isinstance(landmarks, dict):
        landmarks = {k: _get_fun(v) for k, v in landmarks.items()}
    else:
        landmarks = {_to_string(fn_str): _get_fun(fn_str)
                     for fn_str in landmarks}

    r = {k: _validate_pos(v(gtf)) for k, v in landmarks.items()}
    _logger.info("Done!")
    return r


def range_start(gtf):
    position = np.where(gtf.strand == "-", gtf.end, gtf.start)
    return pd.DataFrame({"seqname": gtf.seqname,
                         "position": position,
                         "strand": gtf.strand})


def range_end(gtf):
    position = np.where(gtf.strand == "-", gtf.start, gtf.end)
    return pd.DataFrame({"seqname": gtf.seqname,
                         "position": position,
                         "strand": gtf.strand})


# --------------------------------------------
# landmark extractors

def tss(gr):
    return range_start(gr[gr.feature == "transcript"])


def polya(gr):
    return range_end(gr[gr.feature == "transcript"])


def gene_start(gr):
    return range_start(gr[gr.feature == "gene"])


def gene_end(gr):
    return range_end(gr[gr.feature == "gene"])


def exon_intron(gr):
    ei = range_end(gr[gr.feature == "exon"])
    # filter transforitps
    # remove positions that are also on the transcript ends
    gr_polya = polya(gr)
    # set both to indices
    gr_polya.set_index(gr_polya.columns.tolist(), inplace=True)
    ei.set_index(ei.columns.tolist(), inplace=True)
    ei = ei.loc[ei.index.difference(gr_polya.index)].reset_index()
    return ei


def intron_exon(gr):
    ie = range_start(gr[gr.feature == "exon"])
    # filter transripts
    # remove positions that are also on the transcript start
    gr_tss = tss(gr)
    # set both to indices
    gr_tss.set_index(gr_tss.columns.tolist(), inplace=True)
    ie.set_index(ie.columns.tolist(), inplace=True)
    ie = ie.loc[ie.index.difference(gr_tss.index)].reset_index()
    return ie


def start_codon(gr):
    return range_start(gr[gr.feature == "start_codon"])


def stop_codon(gr):
    return range_end(gr[gr.feature == "stop_codon"])
# --------------------------------------------
# def tss(gtffile):
#     # Code from HTSeq
#     tsspos = set()
#     for feature in gtffile:
#         if feature.type == "exon" and feature.attr["exon_number"] == "1":
#             tsspos.add(feature.iv.start_d_as_pos)
#     return tsspos


def get(name):
    if callable(name):
        return name
    else:
        return get_from_module(name, globals())


def _validate_pos(df):
    """Validates the returned positional object
    """
    assert isinstance(df, pd.DataFrame)
    assert ["seqname", "position", "strand"] == df.columns.tolist()
    assert df.position.dtype == np.dtype("int64")
    assert df.strand.dtype == np.dtype("O")
    assert df.seqname.dtype == np.dtype("O")
    return df


def _get_fun(fn_str):
    if isinstance(fn_str, str):
        return get(fn_str)
    elif callable(fn_str):
        return fn_str
    else:
        raise ValueError("fn_str has to be callable or str")

