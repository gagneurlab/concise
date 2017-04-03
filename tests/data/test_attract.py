from concise.data import attract
from concise.utils.pwm import PWM


def test_attract():
    dt = attract.get_metadata()
    pwm_list = attract.get_pwm_list(dt[-5:]["PWM_id"])

    assert isinstance(pwm_list[0], PWM)
    assert dt[-1:]["Motif"].values[0].replace("U", "T") == pwm_list[-1].get_consensus()

    pwm_list_all = attract.get_pwm_list(dt["PWM_id"].unique())
