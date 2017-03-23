# -*- coding: utf-8 -*-

__author__ = 'Žiga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.4.4'

from .concise import Concise, ConciseCV
from .get_data import prepare_data
from .args_sampler import sample_params
from .kmer import best_kmers

# Add all the custom objects to keras
# TODO - include them automatically? - you are overriging
from keras.utils.generic_utils import get_custom_objects
from . import initializers, regularizers, layers, activations

# initializers
get_custom_objects()['PWMKernelInitializer'] = initializers.PWMKernelInitializer
get_custom_objects()['PWMBiasInitializer'] = initializers.PWMBiasInitializer

# regularizers
get_custom_objects()['GAMRegularizer'] = regularizers.GAMRegularizer

# layers
get_custom_objects()['GAMSmooth'] = layers.GAMSmooth
get_custom_objects()['GlobalSumPooling1D'] = layers.GlobalSumPooling1D
get_custom_objects()['ConvDNA'] = layers.ConvDNA
get_custom_objects()['InputDNA'] = layers.InputDNA
get_custom_objects()['InputDNAQuantity'] = layers.InputDNAQuantity
get_custom_objects()['InputDNAPosition'] = layers.InputDNAPosition

# activations
get_custom_objects()['exponential'] = activations.exponential


