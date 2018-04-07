"""Test visualization functionality
"""
import pytest
import numpy as np
from concise.preprocessing import (encodeDNA, encodeSplines,
                                   encodeRNA, encodeAA, encodeCodon,
                                   encodeRNAStructure)
import concise.layers as cl
from concise.layers import ConvDNA, ConvRNA, ConvAA
import concise.initializers as ci
from concise.utils import PWM
from keras.models import Model
import keras.layers as kl
from keras.models import load_model, Model
from concise.data import encode
from concise.utils.plot import heatmap, seqlogo, seqlogo_fig
import matplotlib.pyplot as plt
import numpy as np


def manual_test_heatmap():
    pwm_elem = encode.get_pwm_list(["AFP_1"])[0]
    pwm = pwm_elem.pwm
    pssm = pwm_elem.get_pssm()

    ax = heatmap(pwm.T, 0, 1, diverge_color=False,
                 vocab=["A", "C", "G", "T"], figsize=(6, 2))

    plt.show()
    multiple_pwm = np.stack([pwm.T, pwm.T, pwm.T, pwm.T, pwm.T], axis=2)
    ax = heatmap(multiple_pwm, 0, 1, diverge_color=False, ncol=3, plot_name="",
                 vocab=["A", "C", "G", "T"], figsize=(8, 4))
    plt.show()

    ax = heatmap(pssm.T, -1, 1, diverge_color=False,
                 vocab=["A", "C", "G", "T"])
    plt.show()

    ax = heatmap(pssm.T, -1, 1, diverge_color=False, plot_name="plot",
                 vocab=["A", "C", "G", "T"])
    plt.show()

def manual_test_layer_plots():
    motifs = ["TTAATGA"]
    pwm_list = [PWM.from_consensus(motif) for motif in motifs]
    seq_length = 100
    motif_width = 7
    # specify the input shape
    input_dna = cl.InputDNA(seq_length)

    # convolutional layer with filters initialized on a PWM
    x = ConvDNA(filters=2,
                kernel_size=motif_width,  # motif width
                activation="relu",
                kernel_initializer=ci.PSSMKernelInitializer(pwm_list),
                bias_initializer=ci.PSSMBiasInitializer(pwm_list, kernel_size=motif_width, mean_max_scale=1)
                # mean_max_scale of 1 means that only consensus sequence gets score larger than 0
                )(input_dna)

    # Smoothing layer - positional-dependent effect
    x = cl.GAMSmooth(n_bases=10, l2_smooth=1e-3, l2=0)(x)
    x = cl.GlobalSumPooling1D()(x)
    x = kl.Dense(units=1, activation="linear")(x)
    model = Model(inputs=input_dna, outputs=x)
    model.compile("adam", "mse")
    # TODO - test
    model.layers[1].plot_weights(plot_type="heatmap")

    model.layers[1].plot_weights(0, plot_type="motif_raw")
    model.layers[1].plot_weights(0, plot_type="motif_pwm_info")

def manual_test_layer_plots_RNA():
    motifs = ["TTAATGA"]
    pwm_list = [PWM.from_consensus(motif) for motif in motifs]
    seq_length = 100
    motif_width = 7
    # specify the input shape
    input_dna = cl.InputDNA(seq_length)

    # convolutional layer with filters initialized on a PWM
    x = ConvRNA(filters=1,
                kernel_size=motif_width,  # motif width
                activation="relu",
                kernel_initializer=ci.PSSMKernelInitializer(pwm_list),
                bias_initializer=ci.PSSMBiasInitializer(pwm_list, kernel_size=motif_width, mean_max_scale=1)
                # mean_max_scale of 1 means that only consensus sequence gets score larger than 0
                )(input_dna)

    # Smoothing layer - positional-dependent effect
    x = cl.GAMSmooth(n_bases=10, l2_smooth=1e-3, l2=0)(x)
    x = cl.GlobalSumPooling1D()(x)
    x = kl.Dense(units=1, activation="linear")(x)
    model = Model(inputs=input_dna, outputs=x)
    model.compile("adam", "mse")
    # TODO - test
    model.layers[1].plot_weights(plot_type="heatmap")

    model.layers[1].plot_weights(0, plot_type="motif_raw")
    model.layers[1].plot_weights(0, plot_type="motif_pwm_info")


def manual_test_layer_plots_AA():
    motifs = ["ACDEFGGIKNY"]

    seq = encodeAA(motifs)

    seq_length = 100
    motif_width = 7

    seqlogo_fig(seq[0], vocab="AA")
    plt.show()

    # specify the input shape
    input_dna = cl.InputAA(seq_length)

    # convolutional layer with filters initialized on a PWM
    x = ConvAA(filters=1,
               kernel_size=motif_width,  # motif width
               activation="relu",
               # mean_max_scale of 1 means that only consensus sequence gets score larger than 0
               )(input_dna)

    # Smoothing layer - positional-dependent effect
    x = cl.GAMSmooth(n_bases=10, l2_smooth=1e-3, l2=0)(x)
    x = cl.GlobalSumPooling1D()(x)
    x = kl.Dense(units=1, activation="linear")(x)
    model = Model(inputs=input_dna, outputs=x)
    model.compile("adam", "mse")
    # TODO - test
    model.layers[1].plot_weights(plot_type="heatmap")

    model.layers[1].plot_weights(0, plot_type="motif_raw")
    model.layers[1].plot_weights(0, plot_type="motif_pwm_info")

def manual_test_layer_plots_Codon():
    motifs = ["TTAATGAAT"]
    seq_length = 102
    # specify the input shape
    input_dna = cl.InputCodon(seq_length)

    # convolutional layer with filters initialized on a PWM
    x = cl.ConvCodon(filters=1,
                     kernel_size=2,  # motif width
                     activation="relu",
                     # mean_max_scale of 1 means that only consensus sequence gets score larger than 0
                     )(input_dna)

    # Smoothing layer - positional-dependent effect
    x = cl.GAMSmooth(n_bases=10, l2_smooth=1e-3, l2=0)(x)
    x = cl.GlobalSumPooling1D()(x)
    x = kl.Dense(units=1, activation="linear")(x)
    model = Model(inputs=input_dna, outputs=x)
    model.compile("adam", "mse")
    model.layers[1].plot_weights(figsize=(3, 10))
