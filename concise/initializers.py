from keras import layers as kl
from keras import regularizers as kr
from keras.initializers import Initializer, _compute_fans
from keras import backend as K
import concise
from concise.utils.pwm import PWM, pwm_list2pwm_array, pwm_array2pssm_array, DEFAULT_BASE_BACKGROUND

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


class PSSMBiasInitializer(Initializer):

    def __init__(self, pwm_list=[], kernel_size=None, mean_max_scale=0.,
                 background_probs=DEFAULT_BASE_BACKGROUND):
        """Bias initializer

        By defult, it will initialize all weights to 0.

        # Arguments
            pwm_list: list of PWM's
            kernel_size: Has to be the same as kernel_size in kl.Conv1D
            mean_max_scale: float; factor for convex conbination between
                                    mean pwm match (mean_max_scale = 0.) and
                                    max pwm match (mean_max_scale = 1.)
            background_probs: A dictionary of background probabilities. Default: `{'A': .25, 'C': .25, 'G': .25, 'T': .25}`
        """

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

# TODO - specify the fraction of noise?
# stddev_pwm
# stddev_frac_pssm
#

# scale_glorot feature:
# TODO - add shift_mean_max_scale - this allows you to drop the bias initializer?
#        - how to call this argument better?
# TODO - write some unit tests checking the initialization scale
# TODO - finish the PWM initialization example notebook


# TODO - why glorot normal and not uniform?
# TODO - can we have just a single initializer for both, pwm and pssm?

# IDEA - draw first a dirichlet distributed pwm (sums to 1) and then transform it to the pssm
#        - how to choose the parameters of the dirichlet distribution?
#           - create a histogram of all pwm values (for each base)
#
# related papers: http://web.stanford.edu/~hmishfaq/cs273b.pdf

# alpha * random + (1 - alpha) * motif


class PSSMKernelInitializer(Initializer):
    """Initializer that generates tensors with a
    truncated normal initializer shifted by
    a position specific scoring matrix (PSSM)

    # Arguments
        pwm_list: a list of `concise.utils.pwm.PWM`'s
        stddev: a python scalar or a scalar tensor. Standard deviation of the
    random values to generate.
        seed: A Python integer. Used to seed the random generator.
        background_probs: A dictionary of background probabilities.
    Default: `{'A': .25, 'C': .25, 'G': .25, 'T': .25}`
        scale_glorot: boolean; If True, each generated filter is min-max scaled to match 

    resulting PWM's are centered and rescaled
    to match glorot_normal distribution.
        add_noise_before_Pwm2Pssm: boolean; if True, add random noise before the
    pwm->pssm transformation

    # TODO - write down the exact formula for this initialization
    """

    def __init__(self, pwm_list=[], stddev=0.05, seed=None,
                 background_probs=DEFAULT_BASE_BACKGROUND,
                 scale_glorot=True,
                 add_noise_before_Pwm2Pssm=True):
        if len(pwm_list) > 0 and isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        self.pwm_list = pwm_list
        _check_pwm_list(pwm_list)
        self.stddev = stddev
        self.seed = seed
        self.background_probs = background_probs
        self.add_noise_before_Pwm2Pssm = add_noise_before_Pwm2Pssm
        self.scale_glorot = scale_glorot

    def __call__(self, shape, dtype=None):
        print("shape: ", shape)
        pwm = pwm_list2pwm_array(self.pwm_list, shape, dtype, self.background_probs)

        if self.add_noise_before_Pwm2Pssm:
            # adding noise on the pwm level
            pwm = _truncated_normal(mean=pwm,
                                    stddev=self.stddev,
                                    seed=self.seed)
            stddev_after = 0  # don't need to add any further noise on the PSSM level
        else:
            stddev_after = self.stddev
        # Force sttdev to be 0, because noise already added. May just use tf.Variable(pssm)

        # TODO - could be problematic if any pwm < 0
        pssm = pwm_array2pssm_array(pwm, background_probs=self.background_probs)
        pssm = _truncated_normal(mean=pssm,
                                 stddev=stddev_after,
                                 seed=self.seed)
        if self.scale_glorot:
            # max, min for each motif individually
            min_max_range = pssm.max(axis=1).max(0) - pssm.min(axis=1).min(0)
            # TODO - wrong! [1, 2] range will just get rescaled but not centered
            # i.e. *2 will do : [2, 4] and not [-1, 1]
            alpha = _glorot_uniform_scale(shape) * 2 / min_max_range
            pssm = alpha * pssm

        return K.constant(pssm, dtype=dtype)

    def get_config(self):
        return {
            'pwm_list': [pwm.get_config() for pwm in self.pwm_list],
            'stddev': self.stddev,
            'seed': self.seed,
            'background_probs': self.background_probs,
        }


class PWMBiasInitializer(Initializer):
    # TODO - automatically determined kernel_size

    def __init__(self, pwm_list=[], kernel_size=None, mean_max_scale=0.):
        """Bias initializer

        # Arguments
            pwm_list: list of PWM's
            kernel_size: Has to be the same as kernel_size in kl.Conv1D
            mean_max_scale: float; factor for convex conbination between
                                    mean pwm match (mean_max_scale = 0.) and
                                    max pwm match (mean_max_scale = 1.)
        """
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


# TODO pack everything into a single initializer without the bias init?
class PWMKernelInitializer(Initializer):
    """truncated normal distribution shifted by a PWM

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


# util functions


def _glorot_uniform_scale(shape):
    """Compute the glorot_uniform scale
    """
    fan_in, fan_out = _compute_fans(shape)
    return np.sqrt(2 * 3.0 / max(1., float(fan_in + fan_out)))


AVAILABLE = ["PWMBiasInitializer", "PWMKernelInitializer",
             "PSSMBiasInitializer", "PSSMKernelInitializer"]


def get(name):
    return get_from_module(name, globals())
