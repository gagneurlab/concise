.. highlight:: shell

============
Installation
============

Prerequisites
--------------

1. Python (3.4 or 3.5) with pip (see `Python installation guide`_ and `pip documentation`_)
2. TensorFlow python package (see `TensorFlow installation guide`_ or `Installing Tensorflow on AWS GPU-instance`_)

.. _pip documentation: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _TensorFlow installation guide: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
.. _Installing Tensorflow on AWS GPU-instance: http://max-likelihood.com/2016/06/18/aws-tensorflow-setup/

Stable release
---------------

To install CONCISE, run this command in your terminal:

.. code-block:: console

    $ pip install concise

This is the preferred method to install CONCISE, as it will always install the most recent stable release. 


From sources
--------------------

The sources for CONCISE can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/avsecz/concise

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/avsecz/concise/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/avsecz/concise
.. _tarball: https://github.com/avsecz/concise/tarball/master
