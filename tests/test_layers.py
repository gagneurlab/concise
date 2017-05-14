"""
test layer saving and loading
"""
import pytest
import numpy as np
from concise.preprocessing import (encodeDNA, encodeSplines,
                                   encodeRNA, encodeAA, encodeCodon,
                                   encodeRNAStructure)
import concise.layers as cl
import concise.initializers as ci
import concise.regularizers as cr
import concise.metrics as cm
from concise.utils import PWM
from keras.models import Model, Sequential
import keras.layers as kl
from keras.models import load_model, Model


def test_convDNA(tmpdir):
    motifs = ["TTAATGA"]
    pwm_list = [PWM.from_consensus(motif) for motif in motifs]
    seq_length = 100
    motif_width = 7
    # specify the input shape
    input_dna = cl.InputDNA(seq_length)

    # convolutional layer with filters initialized on a PWM
    x = cl.ConvDNA(filters=1,
                   kernel_size=motif_width,  # motif width
                   activation="relu",
                   kernel_initializer=ci.PWMKernelInitializer(pwm_list),
                   bias_initializer=ci.PWMBiasInitializer(pwm_list, kernel_size=motif_width, mean_max_scale=1)
                   # mean_max_scale of 1 means that only consensus sequence gets score larger than 0
                   )(input_dna)

    # Smoothing layer - positional-dependent effect
    x = cl.GAMSmooth(n_bases=10, l2_smooth=1e-3, l2=0)(x)
    x = cl.GlobalSumPooling1D()(x)
    x = kl.Dense(units=1, activation="linear")(x)
    model = Model(inputs=input_dna, outputs=x)

    # compile the model
    model.compile(optimizer="adam", loss="mse", metrics=[cm.var_explained])

    # filepath = "/tmp/model.h5"
    filepath = str(tmpdir.mkdir('data').join('test_keras.h5'))

    model.save(filepath)
    m = load_model(filepath)
    assert isinstance(m, Model)

def test_convDNA_sequential():
    m = Sequential([
        cl.ConvDNA(filters=1, kernel_size=10, seq_length=100)
    ])
    m.compile("adam", loss="binary_crossentropy")

@pytest.mark.parametrize("seq, encodeSEQ, InputSEQ, ConvSEQ", [
    (["ACTTGAATA"], encodeDNA, cl.InputDNA, cl.ConvDNA),
    (["ACUUGAAUA"], encodeRNA, cl.InputRNA, cl.ConvRNA),
    (["ACTTGAATA"], encodeCodon, cl.InputCodon, cl.ConvCodon),
    (["ACTTGAATA"], encodeRNAStructure, cl.InputRNAStructure, cl.ConvRNAStructure),
    (["ARNBCEQ"], encodeAA, cl.InputAA, cl.ConvAA),
    (np.array([[1, 2, 3, 4, 5]]), encodeSplines, cl.InputSplines, cl.ConvSplines),
])
def test_all_layers(seq, encodeSEQ, InputSEQ, ConvSEQ, tmpdir):
    seq_length = len(seq[0])

    # pre-process
    train_x = encodeSEQ(seq)
    train_y = np.array([[1]])
    print(train_x.shape)

    # build model
    inp = InputSEQ(seq_length=seq_length)
    if ConvSEQ == cl.ConvSplines:
        x = ConvSEQ(filters=1)(inp)
    else:
        x = ConvSEQ(filters=1, kernel_size=1)(inp)
    x = cl.GlobalSumPooling1D()(x)
    m = Model(inp, x)
    m.summary()
    m.compile("adam", loss="mse")

    m.fit(train_x, train_y)

    filepath = str(tmpdir.mkdir('data').join('test_keras.h5'))

    print(tmpdir)
    m.save(filepath)
    m = load_model(filepath)
    assert isinstance(m, Model)


def test_ConvSplines(tmpdir):

    x_pos = np.vstack([np.arange(15), np.arange(15)])
    y = np.arange(2)

    x = encodeSplines(x_pos)

    inl = cl.InputSplines(15, 10)
    o = cl.ConvSplines(1, kernel_regularizer=cr.GAMRegularizer(l2_smooth=.5),
                       )(inl)
    o = cl.GlobalSumPooling1D()(o)

    model = Model(inl, o)
    model.compile("Adam", "mse")
    model.fit(x, y)

    filepath = str(tmpdir.mkdir('data').join('test_keras.h5'))

    # load and save the model
    model.save(filepath)
    m = load_model(filepath)
    assert isinstance(m, Model)
