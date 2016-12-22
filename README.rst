===============================
CONCISE
===============================


.. image:: https://img.shields.io/pypi/v/concise.svg
        :target: https://pypi.python.org/pypi/concise

.. image:: https://img.shields.io/travis/Avsecz/concise.svg
        :target: https://travis-ci.org/Avsecz/concise

.. image:: https://readthedocs.org/projects/concise-bio/badge/?version=latest
        :target: https://concise-bio.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

CONCISE (COnvolutional neural Network for CIS-regulatory Elements) is a model for predicting any quatitative outcome (say mRNA half-life) from cis-regulatory sequence using deep learning. 

* Developed by the Gagneur Lab (computational biology):  https://www.gagneurlab.in.tum.de
* Free software: MIT license
* Documentation: https://concise-bio.readthedocs.io

.. image:: https://github.com/Avsecz/concise/blob/master/concise-figure1.png
	:target: https://raw.githubusercontent.com/Avsecz/concise/master/concise-figure1.png
        :width: 60%
        :align: center
    
Features
--------

* Very simple API
* Serializing the model to JSON
  - allows to analyze the results in any langugage of choice
* Helper function for hyper-parameter random search
* CONCISE uses TensorFlow at its core and is hence able of using GPU computing

Installation
------------

After installing the following prerequisites:

1. Python (3.4 or 3.5) with pip (see `Python installation guide`_ and `pip documentation`_)
2. TensorFlow python package (see `TensorFlow installation guide`_ or `Installing Tensorflow on AWS GPU-instance`_)


install CONCISE using pip:
   
::
   
   pip install concise


.. _pip documentation: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _TensorFlow installation guide: https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
.. _Installing Tensorflow on AWS GPU-instance: http://max-likelihood.com/2016/06/18/aws-tensorflow-setup/

Getting Started
---------------

.. code-block:: python
		
   import pandas as pd
   import concise

   # read-in and prepare the data
   dt = pd.read_csv("./data/pombe_half-life_UTR3.csv")
   
   X_feat, X_seq, y, id_vec = concise.prepare_data(dt,
                                                   features=["UTR3_length", "UTR5_length"],
                                                   response="hlt",
                                                   sequence="seq",
                                                   id_column="ID",
                                                   seq_align="end",
                                                   trim_seq_len=500,
                                                 )

   ######
   # Train CONCISE
   ######
      
   # initialize CONCISE
   co = concise.Concise(motif_length = 9, n_motifs = 2, 
                        init_motifs = ("TATTTAT", "TTAATGA"))

   # train:
   # - on a GPU if tensorflow is compiled with GPU support
   # - on a CPU with 5 cores otherwise
   co.train(X_feat[500:], X_seq[500:], y[500:], n_cores = 5)

   # predict
   co.predict(X_feat[:500], X_seq[:500])

   # get fitted weights
   co.get_weights()
   
   # save/load from a file
   co.save("./Concise.json")
   co2 = Concise.load("./Concise.json")

   ######
   # Train CONCISE in 5-fold cross-validation
   ######

   # intialize
   co3 = concise.Concise(motif_length = 9, n_motifs = 2, 
                         init_motifs = ("TATTTAT", "TTAATGA"))
                         
   cocv = concise.ConciseCV(concise_object = co3)

   # train
   cocv.train(X_feat, X_seq, y, id_vec,
              n_folds=5, n_cores=3, train_global_model=True)

   # out-of-fold prediction
   cocv.get_CV_prediction()
	      
   # save/load from a file
   cocv.save("./Concise.json")
   cocv2 = ConciseCV.load("./Concise.json")
	      


Where to go from here:
----------------------

* See the example file `<scripts/example-workflow.py>`_
* Read the API Documenation https://concise-bio.readthedocs.io/en/latest/documentation.html

