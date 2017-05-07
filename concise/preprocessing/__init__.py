from . import sequence
from . import smooth
from . import structure

# TODO - remove these aliases
from .sequence import (encodeSequence, encodeDNA, encodeRNA,
                       encodeCodon, encodeAA,
                       pad_sequences)
from .smooth import encodeSplines
from .structure import encodeRNAStructure
