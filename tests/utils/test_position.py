"""Test landmarks
"""
import pytest
import pandas as pd
import numpy as np
from concise.utils.position import extract_landmarks, get
from gtfparse import read_gtf


@pytest.fixture
def gtf():
    # save a smaller version of the annotation
    # from concise.preprocessing.landmarks import read_gtf
    # gtf_path = "/s/genomes/human/hg38/GRCh38.p7/gencode.v25.annotation.gtf"
    # gtf = read_gtf(gtf_path)
    # gtf_small = gtf[gtf.seqnames == "chr22"]
    # gtf_small.to_pickle("data/gencode_v25_chr22.gtf.pkl.gz")  # 116k

    return read_gtf("data/gencode.v24.annotation_chr22.gtf.gz")


@pytest.fixture
def gtf_simple():
    gtfs = pd.DataFrame({"seqname": "chr22",
                         "feature": ["exon", "exon", "transcript", "transcript"],
                         "start": [100, 300, 100, 150],
                         "end": [200, 400, 200, 250],
                         "strand": ["+", "-", "+", "-"],
                         })
    return gtfs


def test_extractors(gtf_simple):
    """Test that each extractor does what it's supposed to do
    """
    tssp = get("tss")(gtf_simple)
    assert np.all(tssp.position == [100, 250])
    assert np.all(tssp.strand == ["+", "-"])

    pa = get("polya")(gtf_simple)
    assert np.all(pa.position == [200, 150])
    assert np.all(pa.strand == ["+", "-"])

    ie = get("intron_exon")(gtf_simple)
    # only one as we need to throw it out (due to transcript...)
    assert np.all(ie.position == [400])
    assert np.all(ie.strand == ["-"])

    ei = get("exon_intron")(gtf_simple)
    assert np.all(ei.position == [300])
    assert np.all(ei.strand == ["-"])


def test_extract_landmarks(gtf):
    """Run for all the extractors
    """
    ldm = extract_landmarks(gtf)
    assert isinstance(ldm, dict)
    # validation is already done internally using _validate_pos
