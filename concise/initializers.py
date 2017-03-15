from keras import layers as kl
from keras import regularizers as kr
from keras.initializers import Initializer
from keras import backend as K
import concise
from concise.utils import PWM, pwm_list2array

import numpy as np
# Arguments:
# - init_motifs=["TATTTAT", ..., "ACTAAT"]
# - init_motifs_scale=1
# - init_motif_bias=0
# - init_sd_motif=1e-2

def _check_pwm_list(pwm_list):
    """Check the input validity
    """
    for pwm in pwm_list:
        if not isinstance(pwm, PWM):
            raise TypeError("element {0} of pwm_list is not of type PWM".format(pwm))
    return True


# TODO - add custom_objects to keras when loading concise
# ADD: PWMBiasInitializer

# TODO - have from_config and to_config for PWM
# TODO - update PWM*Initializer's from_config
# TODO - how it the serialization working on python - level? primitive objects can be serialized?

# TODO - generic class PWMInitializerAbs?
class PWMBiasInitializer(Initializer):

    def __init__(self, pwm_list=[], mean_max_scale=0.):
        """Bias initializer

        # Arguments
            pwm_list: list of PWM's
            mean_max_scale: float; factor for convex conbination between
                                    mean pwm match (mean_max_scale = 0.) and
                                    max pwm match (mean_max_scale = 1.)
        """
        # handle pwm_list as a dictionary
        if isinstance(pwm_list[0], dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        self.pwm_list = pwm_list
        self.mean_max_scale = mean_max_scale
        _check_pwm_list(pwm_list)

    def __call__(self, shape, dtype=None):
        # pwm_array
        pwma = pwm_list2array(self.pwm_list,
                              shape=(None, 4, shape[0]),
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
            "mean_max_scale": self.mean_max_scale,
        }

# TODO test serialization

class PWMKernelInitializer(Initializer):
    """truncated normal distribution shifted by a PWM

    # Arguments
        pwm_list: a list of PWM's or motifs
        stddev: a python scalar or a scalar tensor. Standard deviation of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, pwm_list=[], stddev=0.05, seed=None):
        if isinstance(pwm_list, dict):
            pwm_list = [PWM.from_config(pwm) for pwm in pwm_list]

        self.stddev = stddev
        self.seed = seed
        self.pwm_list = pwm_list
        _check_pwm_list(pwm_list)

    def __call__(self, shape, dtype=None):
        return K.truncated_normal(shape,
                                  mean=pwm_list2array(self.pwm_list, shape, dtype),
                                  stddev=self.stddev,
                                  dtype=dtype, seed=self.seed)

    def get_config(self):
        return {
            'pwm_list': [pwm.get_config() for pwm in self.pwm_list],
            'stddev': self.stddev,
            'seed': self.seed,
        }
