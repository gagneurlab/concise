from . import sequence
from . import splines
from . import structure

# TODO - remove these aliases
from .sequence import (encodeSequence, encodeDNA, encodeRNA,
                       encodeCodon, encodeAA,
                       pad_sequences)
from .splines import encodeSplines, EncodeSplines
from .structure import encodeRNAStructure
