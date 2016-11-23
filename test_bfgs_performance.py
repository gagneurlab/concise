# test bfgs performance

import concise

# read in json data
import pandas as pd
import os
import json
import importlib.util as iut
from pprint import pprint
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from imp import reload

json_config = "~/projects-work/deepcis/data-offline/jun-pombe/benchmark_metadata.json"

with open(os.path.expanduser(json_config)) as json_file:
    data_config = json.load(json_file)

param = {"pooling_layer": "sum",
         "nonlinearity": "exp",  # relu or exp
         # positional bias
         "n_splines": 10,  # number of b-splines to use. None means don't use pos bias
         "spline_exp": False,
         "share_splines": False,   # should the positional bias be shared among different motifs
         # parameters used by concise.prepare_data
         "trim_seq_len": 1000,  # how many bp of the sequence should we use, None means whole
         "seq_align": "end",   # how to align sequences
         # fitting parameteres
         "batch_size": 1000,
         "n_epochs": 10,
         "n_iterations_checkpoint": 40,
         "step_size": 0.000904,  # Learning rate - Important hyper-parameter
         "step_epoch": 10,
         "step_decay": 0.917,
         # regularization
         "lamb": 2.55e-07,
         "motif_lamb": 1.37e-06,
         "spline_lamb": 4.7e-05,
         "spline_param_lamb": 1.43e-05,
         # weights scale (alternative to defining multiple different step_size)
         "nonlinearity_scale_factor": 0.0925,
         "init_motifs_scale": 0.852,
         "standardize_features": True,
         "regress_out_feat": False,
         # initialization
         "init_motif_bias": None,   # None means use the offset method
         "init_sd_motif": 0.00549,
         "init_sd_w": 9.36e-05,
         "print_every": 1,
}
param["init_motifs"] = tuple(data_config["motifs"])
param["n_motifs"] = len(data_config["motifs"]) + 1  # +1 = background motif
param["motif_length"] = max([len(m) for m in data_config["motifs"]])

path = "~/projects-work/deepcis/" + data_config["data_path"]
dt = pd.read_csv(path)
param['features'] = data_config["additional_features"]  # columns used as additional features
X_feat, X_seq, y, id_vec = concise.prepare_data(dt,
                                                features=param['features'],
                                                response=data_config["response"],
                                                sequence=data_config["sequence"],
                                                id_column=data_config["id"],
                                                seq_align=param['seq_align'],
                                                trim_seq_len=param['trim_seq_len']
)

if param["standardize_features"]:
    X_feat = preprocessing.scale(X_feat)

cocv = concise.concise.ConciseCV(concise_model=concise.Concise(**param))
cocv.train(X_feat, X_seq, y, id_vec,
           n_folds=data_config["n_folds"], n_cores=1,
           train_global_model=True)

# cocv.save("/s/project/deepcis/pombe_benchmark/results/concise_dont_regressout_bfgs_best_model_random.json")
# cocv.save("/s/project/deepcis/pombe_benchmark/results/concise_dont_regressout_bfgs_best_model_known_motifs.json")
cocv.save("/s/project/deepcis/pombe_benchmark/results/concise_dont_regressout_bfgs_best_model_known_motifs_exp.json")
