import pytest

from concise.utils import PWM, pwm_list2pwm_array
import numpy as np

# PWM


def test_pwm_basic():
    pwm = np.array([[1, 2, 3, 4], [2, 4, 4, 5]])
    p = PWM(pwm, name="mypwm")
    assert p.get_consensus() == "TT"


def test_pwm_errors():
    with pytest.raises(Exception):
        pwm = np.array([[-1, 2, 3, 4], [2, 4, 4, 5]])
        PWM(pwm)
    with pytest.raises(Exception):
        pwm = np.array([[0, 0, 0, 0], [2, 4, 4, 5]])
        PWM(pwm)


def test_pwm_from_consensus():
    cons = "TATTTAT"
    p = PWM.from_consensus(cons, background_proportion=0.1)
    assert p.get_consensus() == cons
    assert p.pwm[0, 3] == 0.9


def test_pwm_change_length():
    cons = "TATTTAT"
    p = PWM.from_consensus(cons, background_proportion=0.1)

    p._change_length(3)
    assert p.pwm.shape[0] == 3

    p._change_length(10)
    assert p.pwm.shape[0] == 10

    p._change_length(11)
    assert p.pwm.shape[0] == 11

    p._change_length(10)
    assert p.pwm.shape[0] == 10

    assert np.all(p.pwm[0, :] == .25)

    # correct normalization
    assert np.all(p.pwm.sum(axis=1) == 1)


def test_pwm_list2pwm_array():
    # shape: (kernel_size, 4, filters)
    pwm_list = [PWM(np.array([[1, 2, 3, 4], [2, 4, 4, 5]])),
                PWM(np.array([[1, 2, 1, 4], [2, 10, 4, 5]]))]

    # good defaults
    assert np.array_equal(pwm_list2pwm_array(pwm_list), np.stack([pwm.pwm for pwm in pwm_list], axis=-1))

    # smaller number of pwm's
    assert np.array_equal(pwm_list2pwm_array(pwm_list, (None, 4, 1)), pwm_list[0].pwm.reshape([-1, 4, 1]))

    # correct shapes
    assert pwm_list2pwm_array(pwm_list, shape=(10, 4, 3)).shape == (10, 4, 3)
    assert pwm_list2pwm_array(pwm_list, shape=(2, 4, 50)).shape == (2, 4, 50)
    assert pwm_list2pwm_array(pwm_list, shape=(1, 4, 5)).shape == (1, 4, 5)
    assert pwm_list2pwm_array(pwm_list, shape=(11, 4, 128)).shape == (11, 4, 128)

    # correct type
    pwma = pwm_list2pwm_array(pwm_list, shape=(2, 4, 50), dtype="float32")
    assert pwma.dtype == np.dtype("float32")

    # empty array
    pwm_list = []
    a = pwm_list2pwm_array(pwm_list, (10, 4, 15))
    assert np.all(a == np.ones_like(a) * 0.25)

def test_pwm_pssm():
    pwm_list = [PWM(np.array([[1, 2, 3, 4], [2, 4, 4, 5]])),
                PWM(np.array([[1, 2, 1, 4], [2, 10, 4, 5]]))]

    pssm_list = [pwm.get_pssm() for pwm in pwm_list]
    assert isinstance(pssm_list[0], np.ndarray)
    assert pssm_list[0].shape == pwm_list[0].pwm.shape
############################################
