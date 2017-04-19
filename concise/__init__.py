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
# TODO - include them automatically? - you are overriging
from keras.utils.generic_utils import get_custom_objects

# initializers
get_custom_objects()['PWMKernelInitializer'] = initializers.PWMKernelInitializer
get_custom_objects()['PWMBiasInitializer'] = initializers.PWMBiasInitializer
get_custom_objects()['PSSMKernelInitializer'] = initializers.PSSMKernelInitializer
get_custom_objects()['PSSMBiasInitializer'] = initializers.PSSMBiasInitializer

# regularizers
get_custom_objects()['GAMRegularizer'] = regularizers.GAMRegularizer

# layers
get_custom_objects()['GAMSmooth'] = layers.GAMSmooth
get_custom_objects()['GlobalSumPooling1D'] = layers.GlobalSumPooling1D
get_custom_objects()['ConvDNA'] = layers.ConvDNA
get_custom_objects()['ConvDNAQuantitySplines'] = layers.ConvDNAQuantitySplines
get_custom_objects()['InputDNA'] = layers.InputDNA
get_custom_objects()['InputDNAQuantity'] = layers.InputDNAQuantity
get_custom_objects()['InputDNAQuantitySplines'] = layers.InputDNAQuantitySplines

# activations
get_custom_objects()['exponential'] = activations.exponential

# metrics
get_custom_objects()['var_explained'] = metrics.var_explained

# optimizers
get_custom_objects()['AdamWithWeightnorm'] = optimizers.AdamWithWeightnorm
get_custom_objects()['SGDWithWeightnorm'] = optimizers.SGDWithWeightnorm

# remove variables from the scope
del get_custom_objects
