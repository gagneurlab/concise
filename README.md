<div align="center">
    <img src="docs/img/concise_logo_text.jpg" alt="Concise logo" height="64" width="64">
</div>


# Concise: Keras extension for regulatory genomics

[![Build Status](https://travis-ci.org/gagneurlab/concise.svg?branch=master)](https://travis-ci.org/gagneurlab/concise)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fchollet/keras/blob/master/LICENSE)

## 

Concise (originally CONvolutional neural networks for CIS-regulatory Elements) allows you to:

1. Pre-process sequence-related data (`concise.preprocessing`)
    - convert a list of sequences into one-hot-encoded numpy array or tokens.
2. Specify a Keras model with additional modules
    - Concise provides custom `layers`, `initializers` and `regularizers`.
3. Tune the hyper-parameters (`concise.hyopt`)
    - Concise provides convenience functions for working with the `hyperopt` package.
4. Interpret the model
    - most of Concise layers contain plotting methods
5. Share and re-use models
    - every component (layer, initializer, regularizer, loss) is fully compatible with Keras. Model saving and loading works out-of-the-box.


## Installation

Concise is available for Python versions greater than 3.4 and can be installed from [PyPI](pypi.python.org) using `pip`:

```sh
pip install concise
```

To successfully use concise plotting functionality, please also install the libgeos library required by the `shapely` package:

- Ubuntu: `sudo apt-get install -y libgeos-dev`
- Red-hat/CentOS: `sudo yum install geos-devel`

<!-- Make sure your Keras is installed properly and configured with the backend of choice. -->

## Documentation

- <https://i12g-gagneurweb.in.tum.de/public/docs/concise/>


