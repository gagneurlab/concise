"""Train the models
"""
from keras.callbacks import EarlyStopping, History
from hyperopt.mongoexp import MongoTrials
from concise.utils.helper import write_json, merge_dicts
from concise.optimizers import data_based_init
from datetime import datetime
from uuid import uuid4
from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
from copy import deepcopy
import os

# TODO - have a system-wide config for this
DEFAULT_IP = "ouga03"
DEFAULT_SAVE_DIR = "/s/project/deepcis/hyperopt/"


def _put_first(df, names):
    df = df.reindex(columns=names + [c for c in df.columns if c not in names])
    return df


class CMongoTrials(MongoTrials):

    def __init__(self, db_name, exp_name, ip=DEFAULT_IP, port=1234, **kwargs):
        """
        Concise Mongo trials. Extends MonoTrials with the following four methods:

        - valid_tid
        - train_history
        - get_ok_results
        - as_df
        """
        super(CMongoTrials, self).__init__(
            'mongo://{ip}:{p}/{n}/jobs'.format(ip=ip, p=port, n=db_name), exp_key=exp_name, **kwargs)

    def valid_tid(self):
        """List all valid tid's
        """
        return [t["tid"] for t in self.trials if t["result"]["status"] == "ok"]

    def train_history(self, tid=None):
        """Get train history as pd.DataFrame
        """
        def listify(arg):
            if hasattr(type(arg), '__len__'):
                return arg
            return [arg, ]

        def result2history(result):
            return pd.DataFrame(result["history"]["loss"])

        # use all
        if tid is None:
            tid = self.valid_tid()

        res = [result2history(t["result"]).assign(tid=t["tid"]) for t in self.trials if t["tid"] in listify(tid)]
        df = pd.concat(res)
        df = _put_first(df, ["tid"])
        return df

    def get_ok_results(self, verbose=True):
        """Return a list of results with ok status
        """
        not_ok = np.where(np.array(self.statuses()) != "ok")[0]

        if len(not_ok) > 0 and verbose:
            print("{0}/{1} trials were not ok.".format(len(not_ok), len(self.trials)))
            print("Trials: " + str(not_ok))
            print("Statuses: " + str(np.array(self.statuses())[not_ok]))

        r = [merge_dicts({"tid": t["tid"]}, t["result"].to_dict()) for t in self.trials if t["result"]["status"] == "ok"]
        return r

    def as_df(self, ignore_vals=["history"], separator=".", verbose=True):
        """Return a pd.DataFrame view of the whole experiment
        """
        def delete_key(dct, key):
            c = deepcopy(dct)
            assert isinstance(key, list)
            for k in key:
                c.pop(k)
            return c

        def flatten_dict(dd, separator='_', prefix=''):
            return {prefix + separator + k if prefix else k: v
                    for kk, vv in dd.items()
                    for k, v in flatten_dict(vv, separator, kk).items()
                    } if isinstance(dd, dict) else {prefix: dd}

        def add_eval(res):
            if "eval" not in res:
                res["eval"] = {k: v[-1] for k, v in res["history"]["loss"].items()}
            return res

        results = self.get_ok_results(verbose=verbose)
        rp = [flatten_dict(delete_key(add_eval(x), ignore_vals), separator) for x in results]
        df = pd.DataFrame.from_records(rp)

        first = ["tid", "loss", "status"]
        return _put_first(df, first)


class CompileFN():

    def __init__(self, db_name, exp_name,  # TODO - check if we can somehow get those from hyperopt
                 data_module=None, data_name="data",
                 model_module=None, model_name="model",
                 save_dir=DEFAULT_SAVE_DIR,
                 save_model=True,
                 save_results=True,
                 ):
        if not data_module:
            import data as data_module
        if not model_module:
            import models as model_module
        self.data_fun = data_module.get(data_name)
        self.model_fun = model_module.get(model_name)
        self.loss_fun = model_module.get_loss(model_name)
        self.update_param_fun = model_module.get_update_param(model_name)

        self.data_name = data_name
        self.model_name = model_name
        self.db_name = db_name
        self.exp_name = exp_name
        self.save_dir = save_dir
        self.save_model = save_model
        self.save_results = save_results

    def __call__(self, param):
        time_start = datetime.now()

        # get data
        print("load data")
        train, valid, _ = self.data_fun(**merge_dicts(param["data"], param.get("shared", {})))
        time_data_loaded = datetime.now()

        # compute the sequence length etc
        param = self.update_param_fun(param, train)

        # get model
        model = self.model_fun(**merge_dicts(param["model"], param.get("shared", {})))

        # set default early-stop parameters
        if param.get("fit") is None:
            param["fit"] = {}
        if param["fit"].get("epochs") is None:
            param["fit"]["epochs"] = 500
        if param["fit"].get("patience") is None:
            param["fit"]["patience"] = 10

        # weightnorm
        if param["model"].get("use_weightnorm", False):
            # initialize on a batch of data
            # TODO - restrict only to a fraction of the training set
            data_based_init(model, train[0])

        # train the model
        print("fit")
        history = History()
        model.fit(train[0], train[1],
                  validation_data=valid,
                  epochs=param["fit"]["epochs"],
                  verbose=2,
                  callbacks=[history, EarlyStopping(patience=param["fit"]["patience"])])
        time_train_end = datetime.now()

        # evaluate the model
        print("evaluate")
        eval_metrics = model.evaluate(valid[0], valid[1])
        loss = self.loss_fun(eval_metrics)

        # setup paths for storing the data
        # TODO - check if we can somehow get the id from hyperopt
        rid = str(uuid4())
        tm_dir = self.save_dir + "/{db}/{exp}/train_models/".format(db=self.db_name, exp=self.exp_name)
        os.makedirs(tm_dir, exist_ok=True)

        model_path = tm_dir + "{0}.h5".format(rid) if self.save_model else ""
        results_path = tm_dir + "{0}.json".format(rid) if self.save_results else ""

        time_end = datetime.now()
        ret = {"loss": loss,
               "status": STATUS_OK,
               # additional info
               "param": param,
               "path": {
                   "model": model_path,
                   "results": results_path,
               },
               "name": {
                   "data": self.data_name,
                   "model": self.model_name,
               },
               "history": {"params": history.params,
                           "loss": merge_dicts({"epoch": history.epoch}, history.history),
                           },
               # execution times
               "time": {
                   "start": str(time_start),
                   "end": str(time_end),
                   "duration": {
                       "total": (time_end - time_start).total_seconds(),  # in seconds
                       "dataload": (time_data_loaded - time_start).total_seconds(),
                       "training": (time_train_end - time_data_loaded).total_seconds(),
                   }}}

        # optionally save information to disk
        if model_path:
            model.save(model_path)
        if results_path:
            write_json(ret, results_path)
        return ret

    # Style guide:
    # -------------
    #
    # path structure:
    # /s/project/deepcis/hyperopt/db/exp/...
    #                                   /train_models/
    #                                   /best_model.h5

    # hyper-params format:
    #
    # data: ... (pre-preprocessing parameters)
    # model: (architecture, etc)
    # train: (epochs, patience...)
