# **TODO***

- how to nicely show and compile a jupiter notebook as markdown?
- [ ] This notebook (add to issues):
  - get the DeepBind data - maybe use RNACompete?
    - show one example
    - write down where to get this notebook (concise/nbs/getting_started.ipynb)
- [ ] (add to issues) Notebook: hyper-parameter optimization
  - quick-introduction to hyperopt (write as a blog post?)
  - how to use it with keras
  - how to deploy it
  - sample hyper-param optimizer 
- [ ] (add to issues) Notebook: RBP model - show the external positional effeect

--------------------------------------------
# Getting started with concise

## Become familiar with Keras

In order to successfully use concise, please make sure you are familiar with keras. I strongly advise everyone to read the excellent [keras documentation](http://keras.io) first.

Being a keras extension, concise namely tightly follows the keras API.

## DeepBind model in concise

DeepBind model is extremely straightforward.

It's main architecture:

- conv
- maxpool
- dense
- dense

can be expressed in concise in the following way. Note that I prefer to use the functional API of keras.


```python
import concise.layers as cl
import keras.layers as kl
from concise.preprocessing import encodeDNA
from keras.models import Model, load_model

# get the data
seq_list = read_fasta - TODO
seq_list[:5]
y = ...as_matrix()

# encode sequences as one-hot encoded arrays
x_seq = encodeDNA(seq_list)

# specify the model
in_dna = cl.InputDNA(seq_length=100, name="seq")
x = cl.ConvDNA(filters=15, kernel_width=12, activation="relu")(in_dna)
x = kl.MaxPool1D(pool_size=4)(x) # TODO - check
x = kl.Flatten()(x)
x = kl.Dense(100, activation="relu")(x)
out = kl.Dense(1, activation="sigmoid")(x)
m = Model(in_dna, out)
m.compile("adam", loss="binary_crossentropy", metrics=["acc"])

# train the model
m.fit(x_seq, y, epochs=3)

# save the model
m.save("/tmp/model.h5")

# load the model
m2 = load_model(m)

# visualize the filters
m2.layers[1].plot_weights()

m2.layers[1].plot_weights("pwm_info")
```


## Initializing filters on known motifs

In the scenario where data is scarse, it is often useful to initialize the filters to some known position weights matrices (PWM's). That way, the model already starts with a parameter configuration much closes to the 'right' solution.

Concise provides access to (TODO how many) transcription factor PWM's from ENCODE and rna-binding proteins PWM's from ATtrACT (1000 - TODO - show numbers).

### List all the motifs

```python
from concise.data import attract

dfa = attract.get_metadata()
dfa
```

Let's choose a PWMs of the following transcription factors:

```python
from concise.data import attract

dfa = attract.get_metadata()
dfa

dfa.name.contains(["asdas", "asdasdas"])
```

```python
from concise.utils.pwm import PWM
pwm_list = get_pwm_list([123, 3213, 312])
for pwm in pwm_list:
    pwm.plotPWM()
```



Initializing on the known-PWM's and training the model:

...

Show again the PWM's

Note that the with initialization, the filters can be easier interpreted as motifs.


--------------------------------------------

- TODO - show how to do this in concise easily

- write the data function
- write the model function
