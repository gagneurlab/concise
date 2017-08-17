<div align="center">
    <img src="docs/img/concise_logo_text.jpg" alt="Concise logo" height="64" width="64">
</div>


# Concise: Keras extension for regulatory genomics

## 

Concise (CONvolutional neural networks for CIS-regulatory Elements) is a Keras extension for regulatory genomics. 

If allows you to:

1. pre-process sequence-related data (say convert a list of sequences into one-hot-encoded numpy arrays)
2. specify a keras model with additional utilites: concise provides custom `layers`, `initializers` and `regularizers` useful for regulatory genomics
3. tune the hyper-parameters (`hyopt`): concise provides convenience functions for working with `hyperopt` package.
4. interpret: concise layers contain visualization methods
5. share and re-use models: every concise component (layer, initializer, regularizer, loss) is fully compatible with keras:
    -  saving, loading and reusing the models works out-of-the-box

<!-- TODO - include image of concise -->


## Installation

Concise is available for python versions greater than 3.4 and can be installed from source using pip:<!-- PyPI using `pip`: -->

shapely package requires libgeos library. Install it with:

- Ubuntu: `sudo apt-get install -y libgeos-dev`
- Red-hat/CentOS: `sudo yum install geos-devel`


```sh
git clone https://github.com/gagneurlab/concise
pip install --process-dependency-links concise/
```

`--process-dependency-links` and cloning the source is required in order to properly install the following github packages: [deeplift](https://github.com/kundajelab/deeplift) and [simdna](https://github.com/kundajelab/simdna/tarball/0.2#egg=simdna-0.2). PyPI version is also available, but the github packages might not get installed properly:

```sh
pip install concise
```

<!-- Make sure your keras is installed properly and configured with the backend of choice. -->

## Documentation

- <https://i12g-gagneurweb.in.tum.de/public/docs/concise/>


