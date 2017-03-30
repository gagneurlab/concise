# test quasi X
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
from functions.tests import concise_load_data as ld
from functions import concise
from functions import get_data
import pandas as pd
import numpy as np
from imp import reload
from pprint import pprint
pd.set_option("display.max_columns", 100)

# get the data
param, X_feat, X_seq, y, id_vec = ld.load_example_data()

# no positional bias

# TODO: fix the issue with computing the gradient
# - why does it compute the
def test_correct_spline_fit():
    # test the nice print:
    dc = concise.Concise(n_epochs=50, n_splines=10, **param)
    dc.train(X_feat, X_seq, y, X_feat, X_seq, y, n_cores=3)

    # why is the derivative equal to 0?
    w = dc.get_weights()
    w["spline_quasi_X"][0][0]
    w["spline_quasi_X"][0][1]

    # why not y, n_splines ?!:
    y.shape[0], 10

    dc.get_accuracy()

    dc.print_weights()
    # dc.plot_accuracy()
    # dc.plot_pos_bias()
    y_pred = dc.predict(X_feat, X_seq)
    y_pred

