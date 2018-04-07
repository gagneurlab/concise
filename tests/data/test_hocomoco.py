from concise.data import hocomoco
from concise.utils.pwm import PWM


def test_hocomoco():
    dt = hocomoco.get_metadata()
    pwm_list = hocomoco.get_pwm_list(dt[-5:]["PWM_id"])

    assert isinstance(pwm_list[0], PWM)
    assert dt[-1:]["consensus"].values[0] == pwm_list[-1].get_consensus()

    pwm_list_all = hocomoco.get_pwm_list(dt["PWM_id"].unique())
