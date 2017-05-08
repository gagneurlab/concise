"""Test visualization functionality
"""
from concise.data import encode
from concise.utils.plot import heatmap
import matplotlib.pyplot as plt
import numpy as np


def manual_test_heatmap():
    pwm_elem = encode.get_pwm_list(["AFP_1"])[0]
    pwm = pwm_elem.pwm
    pssm = pwm_elem.get_pssm()

    ax = heatmap(pwm.T, 0, 1, diverge_color=False,
                 vocab=["A", "C", "G", "T"])
    plt.show()

    ax = heatmap(pssm.T, -1, 1, diverge_color=False,
                 vocab=["A", "C", "G", "T"])
    plt.show()

    ax = heatmap(pssm.T, -1, 1, diverge_color=False, plot_name="plot",
                 vocab=["A", "C", "G", "T"])
    plt.show()
