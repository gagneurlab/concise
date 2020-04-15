# -*- coding: utf-8 -*-

__author__ = 'Žiga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.6.9'

from . import layers
from . import preprocessing
from . import utils
from . import data
from . import initializers
from . import losses
from . import metrics
from . import eval_metrics
from . import regularizers
from . import hyopt
from . import optimizers

from .legacy.get_data import prepare_data

# Add all the custom objects to keras
from keras.utils.generic_utils import get_custom_objects
custom_objects_modules = [initializers, metrics, regularizers, layers,
                          losses, optimizers]
for mod in custom_objects_modules:
    for f in mod.AVAILABLE:
        get_custom_objects()[f] = mod.get(f)

# remove variables from the scope
del get_custom_objects


# Setup logging
import logging

log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('concise')
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)
