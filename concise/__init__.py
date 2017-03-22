# -*- coding: utf-8 -*-

__author__ = 'Žiga Avsec'
__email__ = 'avsec@in.tum.de'
__version__ = '0.4.4'

from .concise import Concise, ConciseCV
from .get_data import prepare_data
from .args_sampler import sample_params
from .kmer import best_kmers

# Add all the custom objects to keras

from keras.utils.generic_utils import get_custom_objects
from .initializers import PWMKernelInitializer, PWMBiasInitializer
from .layers import GAMSmooth, GlobalSumPooling1D
from .regularizers import GAMRegularizer
from .activations import exponential

get_custom_objects()['PWMKernelInitializer'] = PWMKernelInitializer
get_custom_objects()['PWMBiasInitializer'] = PWMBiasInitializer
get_custom_objects()['GAMSmooth'] = GAMSmooth
get_custom_objects()['GAMRegularizer'] = GAMRegularizer
get_custom_objects()['GlobalSumPooling1D'] = GlobalSumPooling1D
get_custom_objects()['exponential'] = exponential
