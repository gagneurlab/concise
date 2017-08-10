from concise.utils.pwm import PWM
from concise.data import encode


def test_encode():
    dt = encode.get_metadata()
    pwm_list = encode.get_pwm_list(dt[-5:]["PWM_id"])

    assert isinstance(pwm_list[0], PWM)
    assert dt[-1:]["consensus"].values[0] == pwm_list[-1].get_consensus()

    pwm_list_all = encode.get_pwm_list(dt["PWM_id"].unique())
