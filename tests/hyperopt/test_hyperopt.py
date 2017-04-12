import concise
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials
import os
import sys
import time

import subprocess
from tests.hyperopt import data, model
import py.test


# TODO - lacking more unit-tests for CMongoTrials

def manual_test_hyperopt(tmpdir):

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
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    # wait a bit
    time.sleep(1)

    db_name = "test"
    exp_name = "test2"

    fn = concise.hyperopt.CompileFN(db_name, exp_name,
                                    data_module=data, data_name="data",
                                    model_module=model, model_name="build_model",
                                    save_dir=results_path)
    hyper_params = {
        "data": {},
        "shared": {"max_features": 100, "maxlen": 20},
        "model": {"filters": hp.choice("m_filters", (2, 5)),
                  "hidden_dims": 3,
                  },
        "fit": {"epochs": 1}
    }
    # trials = Trials()
    # best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)

    trials = MongoTrials('mongo://localhost:22334/' + db_name + "/jobs", exp_key=exp_name)

    best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=4)
    mongo_worker_proc.terminate()
    mongodb_proc.terminate()

    assert isinstance(best, dict)
    assert "m_filters" in best
