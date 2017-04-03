from concise.data import encode
from concise.utils.pwm import PWM


def test_attract():
    dt = encode.get_metadata()
    pwm_list = encode.get_pwm_list(dt[-5:]["motif_name"])

    assert isinstance(pwm_list[0], PWM)
    assert dt[-1:]["consensus"].values[0].replace("U", "T") == pwm_list[-1].get_consensus()

    pwm_list_all = encode.get_pwm_list(dt["motif_name"].unique())
