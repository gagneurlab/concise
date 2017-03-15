import pytest

from concise.utils import PWM, pwm_list2array
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


def test_pwm_list2array():
    pwm_list = [PWM(np.array([[1, 2, 3, 4], [2, 4, 4, 5]])),
                PWM(np.array([[1, 2, 1, 4], [2, 10, 4, 5]]))]
    assert pwm_list2array(pwm_list, shape=(10, 3)).shape == (10, 3, 4)

    assert pwm_list2array(pwm_list, shape=(2, 3)).shape == (2, 3, 4)
    assert pwm_list2array(pwm_list, shape=(1, 3)).shape == (1, 3, 4)

    # change type
    pwma = pwm_list2array(pwm_list, shape=(1, 3), dtype="float32")
    assert pwma.dtype == np.dtype("float32")

############################################
