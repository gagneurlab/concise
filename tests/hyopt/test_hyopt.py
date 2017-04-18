from hyperopt import fmin, tpe, hp, Trials
import os
import time

from concise.hyopt import CompileFN, CMongoTrials
from concise.utils.helper import merge_dicts
import subprocess
from tests.hyopt import data, model
from copy import deepcopy


def test_compilefn_train_test_split():
    db_name = "test"
    exp_name = "test2"
    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   eval2loss_fn=model.hyperopt_loss_build_model,
                   # eval
                   valid_split=.5,
                   stratified=False,
                   random_state=True,
                   save_dir="/tmp/")
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    trials = Trials()
    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert isinstance(best, dict)


def test_compilefn_cross_val():
    db_name = "test"
    exp_name = "test2"
    fn = CompileFN(db_name, exp_name,
                   cv_n_folds=3,
                   stratified=False,
                   random_state=True,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   eval2loss_fn=model.hyperopt_loss_build_model,
                   save_dir="/tmp/")
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    trials = Trials()
    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert isinstance(best, dict)


def test_hyopt(tmpdir):
    # get the base dir
    mongodb_path = str(tmpdir.mkdir('mongodb'))
    results_path = str(tmpdir.mkdir('results'))
    # mongodb_path = "/tmp/mongodb_test/"
    # results_path = "/tmp/results/"

    proc_args = ["mongod",
                 "--dbpath=%s" % mongodb_path,
                 "--noprealloc",
                 "--port=22334"]
    print("starting mongod", proc_args)
    mongodb_proc = subprocess.Popen(
        proc_args,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        cwd=mongodb_path,  # this prevented mongod assertion fail
    )

    # wait a bit
    time.sleep(1)
    proc_args_worker = ["hyperopt-mongo-worker",
                        "--mongo=localhost:22334/test",
                        "--poll-interval=0.1"]

    mongo_worker_proc = subprocess.Popen(
        proc_args_worker,
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        env=merge_dicts(os.environ, {"PYTHONPATH": os.getcwd()}),
    )
    # --------------------------------------------

    db_name = "test"
    exp_name = "test2"

    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   eval2loss_fn=model.hyperopt_loss_build_model,
                   save_dir=results_path)
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }

    trials = CMongoTrials(db_name, exp_name, ip="localhost",
                          kill_timeout=5 * 60,
                          port=22334)

    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
    assert len(trials) == 2
    assert isinstance(best, dict)
    assert "m_filters" in best

    # test my custom functions
    trials.as_df()
    trials.train_history(trials.valid_tid()[0])
    trials.train_history(trials.valid_tid())
    trials.get_ok_results()

    # --------------------------------------------
    # cross-validation
    db_name = "test"
    exp_name = "test2_cv"

    fn = CompileFN(db_name, exp_name,
                   data_fn=data.data,
                   model_fn=model.build_model,
                   cv_n_folds=3,
                   eval2loss_fn=model.hyperopt_loss_build_model,
                   save_dir=results_path)

    trials = CMongoTrials(db_name, exp_name, ip="localhost",
                          kill_timeout=5 * 60,
                          port=22334)

    best = fmin(fn, deepcopy(hyper_params), trials=trials, algo=tpe.suggest, max_evals=2)
    assert len(trials) == 2
    assert isinstance(best, dict)
    assert "m_filters" in best

    # test my custom functions
    trials.as_df()
    trials.train_history(trials.valid_tid()[0])
    trials.train_history(trials.valid_tid())
    trials.get_ok_results()

    # --------------------------------------------
    # close
    mongo_worker_proc.terminate()
    mongodb_proc.terminate()
