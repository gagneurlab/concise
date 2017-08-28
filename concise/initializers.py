from keras import layers as kl
from keras import regularizers as kr
import keras.initializers as ki
from keras.initializers import Initializer, serialize, deserialize
from keras import backend as K
import concise
from concise.utils.pwm import PWM, pwm_list2pwm_array, pwm_array2pssm_array, DEFAULT_BASE_BACKGROUND
from keras.utils.generic_utils import get_custom_objects

import numpy as np
from scipy.stats import truncnorm
from concise.utils.helper import get_from_module
# Old Concise arguments:
# - init_motifs=["TATTTAT", ..., "ACTAAT"]
# - init_motifs_scale=1
# - init_motif_bias=0
# - init_sd_motif=1e-2

# TODO - REFACTOR - generic class PWMInitializerAbs?


def _check_pwm_list(pwm_list):
    """Check the input validity
    """
    for pwm in pwm_list:
        if not isinstance(pwm, PWM):
            raise TypeError("element {0} of pwm_list is not of type PWM".format(pwm))
    return True


def _truncated_normal(mean,
                      stddev,
                      seed=None,
                      normalize=True,
                      alpha=0.01):
    ''' Add noise with truncnorm from numpy.
    Bounded (0.001,0.999)
    '''
    # within range ()
    # provide entry to chose which adding noise way to use
    if seed is not None:
        np.random.seed(seed)
    if stddev == 0:
        X = mean
    else:
        gen_X = truncnorm((alpha - mean) / stddev,
                          ((1 - alpha) - mean) / stddev,
                          loc=mean, scale=stddev)
        X = gen_X.rvs() + mean
        if normalize:
            # Normalize, column sum to 1
            col_sums = X.sum(1)
            X = X / col_sums[:, np.newaxis]
    return X


class PSSMKernelInitializer(Initializer):
    """Truncated normal distribution shifted by a position-specific scoring matrix (PSSM)

    # Arguments
        pwm_list: a list of PWM's or motifs
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
        background_probs: A dictionary of background probabilities.
                  Default: `{'A': .25, 'C': .25, 'G': .25, 'T': .25}`
        add_noise_before_Pwm2Pssm: bool, if True the gaussian noise is added
    to the PWM (representing nt probabilities) which is then
    transformed to a PSSM with $log(p_{ij}/b_i)$. If False, the noise is added directly to the
    PSSM.
    """

    def __init__(self, pwm_list=[], stddev=0.05, seed=None,
                 background_probs=DEFAULT_BASE_BACKGROUND,
                 add_noise_before_Pwm2Pssm=True):
        if len(pwm_list) > 0 and isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        self.pwm_list = pwm_list
        _check_pwm_list(pwm_list)
        self.stddev = stddev
        self.seed = seed
        self.background_probs = background_probs
        self.add_noise_before_Pwm2Pssm = add_noise_before_Pwm2Pssm

    def __call__(self, shape, dtype=None):
        # print("PWMKernelInitializer shape: ", shape)

        pwm = pwm_list2pwm_array(self.pwm_list, shape, dtype, self.background_probs)

        if self.add_noise_before_Pwm2Pssm:
            # add noise with numpy truncnorm function
            pwm = _truncated_normal(mean=pwm,
                                    stddev=self.stddev,
                                    seed=self.seed)

            pssm = pwm_array2pssm_array(pwm, background_probs=self.background_probs)

            # Force sttdev to be 0, because noise already added. May just use tf.Variable(pssm)
            # return K.Variable(pssm) # this raise error
            return K.truncated_normal(shape,
                                      mean=pssm,
                                      stddev=0,
                                      dtype=dtype, seed=self.seed)
        else:
            pssm = pwm_array2pssm_array(pwm, background_probs=self.background_probs)
            return K.truncated_normal(shape,
                                      mean=pssm,
                                      stddev=self.stddev,
                                      dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'pwm_list': [pwm.get_config() for pwm in self.pwm_list],
            'stddev': self.stddev,
            'seed': self.seed,
            'background_probs': self.background_probs,
        }


class PSSMBiasInitializer(Initializer):
    """Bias initializer complementary to `PSSMKernelInitializer`

    By defult, it will initialize all weights to 0.

    # Arguments
        pwm_list: list of PWM's
        kernel_size: Has to be the same as kernel_size in kl.Conv1D
        mean_max_scale: float; factor for convex conbination between
                                mean pwm match (mean_max_scale = 0.) and
                                max pwm match (mean_max_scale = 1.)
        background_probs: A dictionary of background probabilities. Default: `{'A': .25, 'C': .25, 'G': .25, 'T': .25}`
    """

    def __init__(self, pwm_list=[], kernel_size=None, mean_max_scale=0., background_probs=DEFAULT_BASE_BACKGROUND):

        # handle pwm_list as a dictionary
        if len(pwm_list) > 0 and isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        if kernel_size is None:
            kernel_size = len(pwm_list)

        _check_pwm_list(pwm_list)
        self.pwm_list = pwm_list
        self.kernel_size = kernel_size
        self.mean_max_scale = mean_max_scale
        self.background_probs = background_probs

    def __call__(self, shape, dtype=None):
        # pwm_array
        # print("PWMBiasInitializer shape: ", shape)
        pwm = pwm_list2pwm_array(self.pwm_list,
                                 shape=(self.kernel_size, 4, shape[0]),
                                 background_probs=self.background_probs,
                                 dtype=dtype)

        pssm = pwm_array2pssm_array(pwm, background_probs=self.background_probs)

        # maximum sequence match
        max_scores = np.sum(np.amax(pssm, axis=1), axis=0)
        mean_scores = np.sum(np.mean(pssm, axis=1), axis=0)

        biases = - (mean_scores + self.mean_max_scale * (max_scores - mean_scores))

        # ret = - (biases - 1.5 * self.init_motifs_scale)
        return biases.astype(dtype)

    def get_config(self):
        return {
            "pwm_list": [pwm.get_config() for pwm in self.pwm_list],
            "kernel_size": self.kernel_size,
            "mean_max_scale": self.mean_max_scale,
            "background_probs": self.background_probs,
        }


class PWMKernelInitializer(Initializer):
    """Truncated normal distribution shifted by a PWM

    # Arguments
        pwm_list: a list of PWM's or motifs
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, pwm_list=[], stddev=0.05, seed=None):
        if len(pwm_list) > 0 and isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        self.stddev = stddev
        self.seed = seed
        self.pwm_list = pwm_list
        _check_pwm_list(pwm_list)

    def __call__(self, shape, dtype=None):
        # print("PWMKernelInitializer shape: ", shape)
        return K.truncated_normal(shape,
                                  mean=pwm_list2pwm_array(self.pwm_list, shape, dtype),
                                  stddev=self.stddev,
                                  dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'pwm_list': [pwm.get_config() for pwm in self.pwm_list],
            'stddev': self.stddev,
            'seed': self.seed,
        }


class PWMBiasInitializer(Initializer):
    """Bias initializer complementary to `PWMKernelInitializer`

    # Arguments
        pwm_list: list of PWM's
        kernel_size: Has to be the same as kernel_size in kl.Conv1D
        mean_max_scale: float; factor for convex conbination between
                                mean pwm match (mean_max_scale = 0.) and
                                max pwm match (mean_max_scale = 1.)
    """
    # TODO - automatically determined kernel_size

    def __init__(self, pwm_list=[], kernel_size=None, mean_max_scale=0.):
        # handle pwm_list as a dictionary
        if len(pwm_list) > 0 and isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        if kernel_size is None:
            kernel_size = len(pwm_list)

        self.pwm_list = pwm_list
        self.kernel_size = kernel_size
        self.mean_max_scale = mean_max_scale
        _check_pwm_list(pwm_list)

    def __call__(self, shape, dtype=None):
        # pwm_array
        # print("PWMBiasInitializer shape: ", shape)
        pwma = pwm_list2pwm_array(self.pwm_list,
                                  shape=(self.kernel_size, 4, shape[0]),
                                  dtype=dtype)

        # maximum sequence match
        max_scores = np.sum(np.amax(pwma, axis=1), axis=0)
        # mean sequence match = 0.25 * pwm length
        mean_scores = np.sum(np.mean(pwma, axis=1), axis=0)

        biases = - (mean_scores + self.mean_max_scale * (max_scores - mean_scores))

        # ret = - (biases - 1.5 * self.init_motifs_scale)
        return biases.astype(dtype)

    def get_config(self):
        return {
            "pwm_list": [pwm.get_config() for pwm in self.pwm_list],
            "kernel_size": self.kernel_size,
            "mean_max_scale": self.mean_max_scale,
        }


AVAILABLE = ["PWMBiasInitializer", "PWMKernelInitializer",
             "PSSMBiasInitializer", "PSSMKernelInitializer"]


def get(name):
    try:
        return ki.get(name)
    except ValueError:
        return get_from_module(name, globals())
