import numpy as np

DEFAULT_LETTER_TO_INDEX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# PWM class
class PWM(object):
    letterToIndex = DEFAULT_LETTER_TO_INDEX
    indexToLetter = dict((letterToIndex[x], x) for x in letterToIndex)
    # can we have two different motifs?

    def __init__(self, pwm, name=None):
        """PWM matrix

        ## Arguments
            pwm: np.array or motif ;
            name: PWM name
            motif: None

        """
        self.pwm = pwm
        self.name = name

        # normalize pwm
        if type(pwm).__module__ != np.__name__:
            raise Exception("pwm needs to by a numpy array")
        if pwm.shape[1] != 4 and len(pwm.shape) == 2:
            raise Exception("pwm needs to be n*4, n is pwm_lenght")
        if np.any(pwm < 0):
            raise Exception("All pwm elements need to be positive")
        if not np.all(np.sum(pwm, axis=1) > 0):
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
    def _background_pwm(cls, length=9, probs={"A": .25, "C": .25, "G": .25, "T": .25}):
        pwm = np.array([[probs[cls.indexToLetter[i]] for i in range(4)]
                        for i in range(length)])
        if length == 0:
            pwm = pwm.reshape([0, 4])

        return pwm

    @classmethod
    def from_background(cls, length=9, name=None, probs={"A": .25, "C": .25, "G": .25, "T": .25}):
        return PWM(cls._background_pwm(length, probs),
                   name=name)

    def _change_length(self, new_length, probs={"A": .25, "C": .25, "G": .25, "T": .25}):
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
            print("add_start: {0}".format(add_start))
            # concatenate two arrays
            pwm_start = self._background_pwm(add_start)
            pwm_end = self._background_pwm(add_end)
            self.pwm = np.concatenate([pwm_start, self.pwm, pwm_end], axis=0)

            self.normalize()

        return self

    # TODO - plot motif

    # TODO - load motif from file

# TODO - fix the shape to be compatible with keras conv-filters
def pwm_list2array(pwm_list, shape, dtype=None):
    n_motifs = len(pwm_list)

    required_n_motifs = shape[0]
    required_motif_len = shape[1]

    # fix n_motifs
    if required_n_motifs > n_motifs:
        add_n_pwm = required_n_motifs - n_motifs
        pwm_list += [PWM.from_background(length=required_motif_len)] * add_n_pwm

    if required_n_motifs < n_motifs:
        print("Removing {0} pwm's from pwm_list".format(n_motifs - required_n_motifs))
        pwm_list = pwm_list[:required_n_motifs]

    # fix motif_len
    pwm_list = [pwm._change_length(required_motif_len) for pwm in pwm_list]

    pwm_array = np.stack([pwm.pwm for pwm in pwm_list])
    pwm_array = pwm_array.astype(dtype)

    return pwm_array
