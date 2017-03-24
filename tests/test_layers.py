"""
test layer saving and loading
"""


from concise.preprocessing import encodeDNA, encodeSplines
import concise.layers as cl
import concise.initializers as ci
import concise.regularizers as cr
import concise.metrics as cm
from concise.utils import PWM
from keras.models import Model
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
