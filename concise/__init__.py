# -*- coding: utf-8 -*-

__author__ = 'Žiga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.5.0'

from . import activations
from . import layers
from . import preprocessing
from . import utils
from . import data
from . import initializers
from . import losses
from . import metrics
from . import eval_metrics
from . import models
from . import regularizers
from . import hyopt
from . import optimizers

from .concise import Concise, ConciseCV
from .legacy.get_data import prepare_data
from .args_sampler import sample_params


# Add all the custom objects to keras
from keras.utils.generic_utils import get_custom_objects
custom_objects_modules = [initializers, metrics, regularizers, layers,
                          activations, losses, optimizers]
for mod in custom_objects_modules:
    for f in mod.AVAILABLE:
        get_custom_objects()[f] = mod.get(f)

# remove variables from the scope
del get_custom_objects
