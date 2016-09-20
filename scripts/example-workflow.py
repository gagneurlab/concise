"""
Example workflow for CONCISE
"""
import numpy as np
import pandas as pd
import uuid
import concise


########
# 1. choose the hyper-paramters
########

# define the hyper-parameter ranges
ARGS = {
    # network architecture
    "pooling_layer": {"sum"},
    "nonlinearity": {"relu"},  # relu or exp

    # motif
    "motif_length": {9},
    "n_motifs": {3},        # number of motifs
    "init_motifs": {('TATTTAT', 'TTAATGA', 'ACTAAT')},  # seed motifs

    # positional bias
    "n_splines": {10},    # number of b-splines to use. None means don't use the positional bias
    "spline_exp": {False},
    "share_splines": {False},   # should the positional bias be shared among different motifs

    # parameters used by concise.prepare_data
    "trim_seq_len": {500},  # how many bp of the sequence should we use, None means whole
    "seq_align": {"end"},   # how to align sequences

    # fitting parameteres
    "batch_size": 32,
    "n_epochs": 50,
    "step_size": [0.001, 0.015],  # Learning rate - Important hyper-parameter
    "step_epoch": 10,
    "step_decay": (0.9, 1),

    # regularization
    "lamb": [1e-7, 1e-3],
    "motif_lamb": [1e-9, 1e-3],
    "spline_lamb": [1e-9, 1e-3],
    "spline_param_lamb": [1e-9, 1e-3],

    # weights scale (alternative to defining multiple different step_size)
    "nonlinearity_scale_factor": [1e-3, 1e3],
    "init_motifs_scale": [1e-2, 1e2],

    # initialization
    "init_bias": {None},   # None means use the offset method
    "init_sd_filter": [1e-3, 1e-1],
    "init_sd_w": [1e-5, 1e-1],
    "print_every": 100
}

# randomly sample a point the hyper-parameter space
param = concise.sample_params(ARGS)


########
# 2. Import the data
########

# mRNA half-life data from S. Pombe (Eser, Wachutka et.al. 2016 MSB):
dt = pd.read_csv("./data/pombe_half-life_UTR3.csv")

param['features'] = ["UTR3_length", "UTR5_length"]  # columns used as additional features

X_feat, X_seq, y, id_vec = concise.prepare_data(dt,
                                                features=param['features'],
                                                response="hlt",  # response variable name
                                                sequence="seq",  # DNA/RNA sequence variable name
                                                id_column="ID",  # ID column name
                                                seq_align=param['seq_align'],
                                                trim_seq_len=param['trim_seq_len']
                                                )


########
# 3. Train the CONCISE model
########

# initialize with parameters defined previously
co = concise.Concise(**param)

# train the model
co.train(X_feat[500:], X_seq[500:], y[500:],     # training data
         X_feat[100:500], X_seq[100:500], y[100:500],  # validation data
         n_cores=3)

# predict for the test-set
co.predict(X_feat[:100], X_seq[:100])

# get fitted weights
co.get_weights()

# save to file
filepath = "Concise_{0}.json".format(uuid.uuid4())
co.save(filepath)

# load from a file
co2 = concise.Concise.load(filepath)


########
# 4. Train the CONCISE model in cross-validation
########

# run the model in cross-validation and save the results

# initialize the object
co = concise.Concise(**param)   # use the same parameters as before
cocv = concise.ConciseCV(concise_model=co)

# train Concise in 5-fold cross-validation
cocv.train(X_feat, X_seq, y, id_vec,
           n_folds=5, n_cores=3, train_global_model=True)


# save to file
filepath_cv = "ConciseCV_{0}.json".format(uuid.uuid4())
cocv.save(filepath_cv)

# load from file
cocv2 = ConciseCV.load(filepath_cv)

# compute the mean-squared error on out-of-fold samples
y_cv_pred = cocv2.get_CV_prediction()
np.mean((y - y_cv_pred)**2)

print("Successful execution!")

