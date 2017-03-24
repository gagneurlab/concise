import numpy as np
import copy

DEFAULT_LETTER_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DEFAULT_INDEX_TO_LETTER = dict((DEFAULT_LETTER_TO_INDEX[x], x) for x in DEFAULT_LETTER_TO_INDEX)
DEFAULT_BASE_BACKGROUND = {"A": .25, "C": .25, "G": .25, "T": .25}


class PWM(object):
    letterToIndex = DEFAULT_LETTER_TO_INDEX
    indexToLetter = DEFAULT_INDEX_TO_LETTER

    def __init__(self, pwm, name=None):
        """PWM matrix

        ## Arguments
            pwm: np.array or motif ;
            name: PWM name
            motif: None

        """
        self.pwm = np.asarray(pwm)  # needs to by np.array
        self.name = name

        # if type(pwm).__module__ != np.__name__:
        #     raise Exception("pwm needs to by a numpy array")
        if self.pwm.shape[1] != 4 and len(self.pwm.shape) == 2:
            raise Exception("pwm needs to be n*4, n is pwm_lenght")
        if np.any(self.pwm < 0):
            raise Exception("All pwm elements need to be positive")
        if not np.all(np.sum(self.pwm, axis=1) > 0):
            raise Exception("All pwm rows need to have sum > 0")

        # normalize the pwm
        self.normalize()

    def normalize(self):
        rows = np.sum(self.pwm, axis=1)
        self.pwm = self.pwm / rows.reshape([-1, 1])

    def get_consensus(self):
        max_idx = self.pwm.argmax(axis=1)
        return ''.join([self.indexToLetter[x] for x in max_idx])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "PWM(name: {0}, consensus: {1})".format(self.name, self.get_consensus())

    @classmethod
    def from_consensus(cls, consensus_seq, background_proportion=0.1, name=None):
        pwm = np.zeros((len(consensus_seq), 4))
        pwm += background_proportion / 3

        for (i, l) in enumerate(consensus_seq):
            b = cls.letterToIndex[l]
            pwm[i, b] = 1 - background_proportion

        return cls(pwm, name=name)

    @classmethod
    def _background_pwm(cls, length=9, probs=DEFAULT_BASE_BACKGROUND):
        pwm = np.array([[probs[cls.indexToLetter[i]] for i in range(4)]
                        for i in range(length)])
        if length == 0:
            pwm = pwm.reshape([0, 4])

        return pwm

    @classmethod
    def from_background(cls, length=9, name=None, probs=DEFAULT_BASE_BACKGROUND):
        return PWM(cls._background_pwm(length, probs),
                   name=name)

    def _change_length(self, new_length, probs=DEFAULT_BASE_BACKGROUND):
        length = self.pwm.shape[0]
        len_diff = new_length - length
        if (len_diff < 0):
            # symmetrically remove
            remove_start = abs(len_diff) // 2
            remove_end = abs(len_diff) // 2 + abs(len_diff) % 2
            self.pwm = self.pwm[remove_start:(length - remove_end), :]

        if (len_diff > 0):
            add_start = len_diff // 2 + len_diff % 2
            add_end = len_diff // 2
            # concatenate two arrays
            pwm_start = self._background_pwm(add_start)
            pwm_end = self._background_pwm(add_end)
            self.pwm = np.concatenate([pwm_start, self.pwm, pwm_end], axis=0)

            self.normalize()

        return self

    def get_config(self):
        return {"pwm": self.pwm.tolist(),  # convert numpyarray to list
                "name": self.name
                }

    @classmethod
    def from_config(cls, pwm_dict):
        return cls(**pwm_dict)

    # TODO - plot motif

    # TODO - load motif from file


def pwm_list2array(pwm_list, shape=(None, 4, None), dtype=None):
    # print("shape: ", shape)
    if shape[1] is not 4:
        raise ValueError("shape[1] has to be 4 and not {0}".format(shape[1]))

    # copy pwm_list
    pwm_list = copy.deepcopy(pwm_list)

    n_motifs = len(pwm_list)

    # set the default values
    shape = list(shape)
    if shape[0] is None:
        # max pwm length
        shape[0] = max([pwm.pwm.shape[0] for pwm in pwm_list])
    if shape[2] is None:
        shape[2] = n_motifs

    # (kernel_size, 4, filters)
    required_motif_len = shape[0]
    required_n_motifs = shape[2]

    # fix n_motifs
    if required_n_motifs > n_motifs:
        add_n_pwm = required_n_motifs - n_motifs
        pwm_list += [PWM.from_background(length=required_motif_len)] * add_n_pwm

    if required_n_motifs < n_motifs:
        print("Removing {0} pwm's from pwm_list".format(n_motifs - required_n_motifs))
        pwm_list = pwm_list[:required_n_motifs]

    # fix motif_len
    pwm_list = [pwm._change_length(required_motif_len) for pwm in pwm_list]

    # stack the matrices along the last axis
    pwm_array = np.stack([pwm.pwm for pwm in pwm_list], axis=-1)
    pwm_array = pwm_array.astype(dtype)

    # change the axis order
    return pwm_array
