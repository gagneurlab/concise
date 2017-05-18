"""
Sample random arguments from a dict of parameters
"""
import numpy as np
import random


def sample_params(params):
    """Randomly sample hyper-parameters stored in a dictionary on a predefined range and scale.

    Useful for hyper-parameter random search.

    Args:
        params (dict): hyper-parameters to sample. Dictionary value-type parsing:

                     - :python:`[1e3, 1e7]` - uniformly sample on a **log10** scale from the interval :python:`(1e3,1e7)`
                     - :python:`(1, 10)` - uniformly sample on a **normal** scale from the interval :python:`(1,10)`
                     - :python:`{1, 2}` - sample from a **set** of values.
                     - :python:`1` - don't sample

    Returns:
        dict: Dictionary with the same keys as :py:attr:`params`, but with only one element as the value.


    Examples:
        >>> myparams = {
            "max_pool": True, # allways use True
            "step_size": [0.09, 0.005],
            "step_decay": (0.9, 1),
            "n_splines": {10, None}, # use either 10 or None
            "some_tuple": {(1,2), (1)},
        }
        >>> concise.sample_params(myparams)
        {'step_decay': 0.9288, 'step_size': 0.0292, 'max_pool': True, 'n_splines': None, 'some_tuple': (1, 2)}
        >>> concise.sample_params(myparams)
        {'step_decay': 0.9243, 'step_size': 0.0293, 'max_pool': True, 'n_splines': None, 'some_tuple': (1)}
        >>> concise.sample_params(myparams)
        {'step_decay': 0.9460, 'step_size': 0.0301, 'max_pool': True, 'n_splines': 10, 'some_tuple': (1, 2)}    

    Note:
        - :python:`{[1,2], [3,4]}` is invalid. Use :python:`{(1,2), (3,4)}` instead.
        - You can allways use :python:`{}` with a single element to by-pass sampling.

    """
    def sample_log(myrange):
        x = np.random.uniform(np.log10(myrange[0]), np.log10(myrange[1]))
        return 10**x

    def sample_unif(myrange):
        x = np.random.uniform(myrange[0], myrange[1])
        return x

    def sample_set(myset):
        x = random.sample(myset, 1)
        return x[0]

    def type_dep_sample(myrange):
        if type(myrange) is list:
            return sample_log(myrange)

        if type(myrange) is tuple:
            return sample_unif(myrange)

        if type(myrange) is set:
            return sample_set(myrange)
        return myrange

    return {k: type_dep_sample(v) for k, v in params.items()}
