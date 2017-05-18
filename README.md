# Concise: Keras extension for regulatory genomics

## 

Concise (CONvolutional neural networks for CIS-regulatory Elements) is a Keras extension for regulatory genomics. 

If provides functions along all the modelling steps:

1. `preprocessing` (convert list of sequences into numpy arrays)
2. specify the keras model: concise provides custom `layers`, `initializers` and `regularizers` useful inregulatory genomics
3. hyper-parameter tuning (`hyopt`): convenience functions for working with `hyperopt` package.
4. interpretation: concise layers contain visualization methods
5. share and re-use models: every concise component (layer, initializer, regularizer, loss) is fully compatible with keras:
    -  saving, loading and reusing the models works out-of-the-box

<!-- TODO - include image of concise -->


## Installation

Concise is available for python versions greater than 3.4 and can be installed from PyPI using `pip`:

```sh
pip install --process-dependency-links concise
```

Note the `--process-dependency-links` is required in order to properly install the following github packages: [deeplift](https://github.com/kundajelab/deeplift) and [simdna](https://github.com/kundajelab/simdna/tarball/0.2#egg=simdna-0.2).

Make sure your keras is installed properly and configured with the backend of choice.
