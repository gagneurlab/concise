from itertools import groupby
from collections import OrderedDict


def read_fasta(file_path):
    """Read the fasta file as `OrderedDict`
    """
    return OrderedDict([x for x in iter_fasta(file_path)])


def iter_fasta(file_path):
    """Returns an iterator over the fasta file

    Given a fasta file. yield tuples of header, sequence

    Code modified from Brent Pedersen's:
    "Correct Way To Parse A Fasta File In Python"


    # Example

        ```python
            fasta = fasta_iter("hg19.fa")
            for header, seq in fasta:
               print(header)
        ```
    """
    fh = open(file_path)

    # ditch the boolean (x[0]) and just keep the header or sequence since
    # we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()
        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())
        yield (headerStr, seq)


def write_fasta(file_path, seq_list, name_list=None):
    """Write a fasta file

    # Arguments
      file_path: file path
      seq_list: List of strings
      name_list: List of names corresponding to the sequences.
    If not None, it should have the same length as `seq_list`
    """
    if name_list is None:
        name_list = [str(i) for i in range(len(seq_list))]

    # needs to be dict or seq
    with open(file_path, "w") as f:
        for i in range(len(seq_list)):
            f.write(">" + name_list[i] + "\n" + seq_list[i] + "\n")
