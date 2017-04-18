"""Train the models
"""
from keras.callbacks import EarlyStopping, History
from hyperopt.utils import coarse_utcnow
from hyperopt.mongoexp import MongoTrials
from concise.utils.helper import write_json, merge_dicts
from concise.utils.model_data import (subset, split_train_test_idx, split_KFold_idx)
from datetime import datetime, timedelta
from uuid import uuid4
from hyperopt import STATUS_OK
import numpy as np
import pandas as pd
from copy import deepcopy
import os
import pprint
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# TODO - have a system-wide config for this
DEFAULT_IP = "ouga03"
DEFAULT_SAVE_DIR = "/s/project/deepcis/hyperopt/"


def _put_first(df, names):
    df = df.reindex(columns=names + [c for c in df.columns if c not in names])
    return df


def _listify(arg):
    if hasattr(type(arg), '__len__'):
        return arg
    return [arg, ]


class CMongoTrials(MongoTrials):

    def __init__(self, db_name, exp_name, ip=DEFAULT_IP, port=1234, kill_timeout=None, **kwargs):
        """
        Concise Mongo trials. Extends MonoTrials with the following four methods:

        - valid_tid
        - train_history
        - get_ok_results
        - as_df

        kill_timeout: After how many seconds to kill the job if it was stalled. None for not killing it
        """
        self.kill_timeout = kill_timeout
        if self.kill_timeout is not None and self.kill_timeout < 60:
            logger.warning("kill_timeout < 60 -> Very short time for " +
                           "each job to complete before it gets killed!")

        super(CMongoTrials, self).__init__(
            'mongo://{ip}:{p}/{n}/jobs'.format(ip=ip, p=port, n=db_name), exp_key=exp_name, **kwargs)

    # def refresh(self):
    #     """Extends the original object
    #     """
    #     self.refresh_tids(None)
    #     if self.kill_timeout is not None:
    #         # TODO - remove dry_run
    #         self.delete_running(self.kill_timeout, dry_run=True)

    def count_by_state_unsynced(self, arg):
        """Extends the original object in order to inject checking
        for stalled jobs in the running jobs
        """
        if self.kill_timeout is not None:
            self.delete_running(self.kill_timeout)
        return super(CMongoTrials, self).count_by_state_unsynced(arg)

    def delete_running(self, timeout_last_refresh=0, dry_run=False):
        """Delete jobs stalled in the running state for too long

        timeout_last_refresh, int: number of seconds
        """
        running_all = self.handle.jobs_running()
        running_timeout = [job for job in running_all
                           if coarse_utcnow() > job["refresh_time"] +
                           timedelta(seconds=timeout_last_refresh)]
        if len(running_timeout) == 0:
            # Nothing to stop
            self.refresh_tids(None)
            return None

        if dry_run:
            logger.warning("Dry run. Not removing anything.")

        logger.info("Removing {0}/{1} running jobs. # all jobs: {2} ".
                    format(len(running_timeout), len(running_all), len(self)))

        now = coarse_utcnow()
        logger.info("Current utc time: {0}".format(now))
        logger.info("Time horizont: {0}".format(now - timedelta(seconds=timeout_last_refresh)))
        for job in running_timeout:
            logger.info("Removing job: ")
            pjob = job.to_dict()
            del pjob["misc"]  # ignore misc when printing
            logger.info(pprint.pformat(pjob))
            if not dry_run:
                self.handle.delete(job)
                logger.info("Job deleted")
        self.refresh_tids(None)

    def valid_tid(self):
        """List all valid tid's
        """
        return [t["tid"] for t in self.trials if t["result"]["status"] == "ok"]

    def train_history(self, tid=None):
        """Get train history as pd.DataFrame
        """

        def result2history(result):
            if isinstance(result["history"], list):
                return pd.concat([pd.DataFrame(hist["loss"]).assign(fold=i)
                                  for i, hist in enumerate(result["history"])])
            else:
                return pd.DataFrame(result["history"]["loss"])

        # use all
        if tid is None:
            tid = self.valid_tid()

        res = [result2history(t["result"]).assign(tid=t["tid"]) for t in self.trials
               if t["tid"] in _listify(tid)]
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
                if isinstance(res["history"], list):
                    # take the average across all folds
                    eval_names = list(res["history"][0]["loss"].keys())
                    eval_metrics = np.array([[v[-1] for k, v in hist["loss"].items()]
                                             for hist in res["history"]]).mean(axis=0).tolist()
                    res["eval"] = {eval_names[i]: eval_metrics[i] for i in range(len(eval_metrics))}
                else:
                    res["eval"] = {k: v[-1] for k, v in res["history"]["loss"].items()}
            return res

        results = self.get_ok_results(verbose=verbose)
        rp = [flatten_dict(delete_key(add_eval(x), ignore_vals), separator) for x in results]
        df = pd.DataFrame.from_records(rp)

        first = ["tid", "loss", "status"]
        return _put_first(df, first)


def _train_and_eval_single(train, valid, model, batch_size=32, epochs=300, callbacks=[]):
    """Fit and evaluate a keras model
    """
    def _format_keras_history(history):
        """nicely format keras history
        """
        return {"params": history.params,
                "loss": merge_dicts({"epoch": history.epoch}, history.history),
                }
    # train the model
    logger.info("Fit...")
    history = History()
    model.fit(train[0], train[1],
              batch_size=batch_size,
              validation_data=valid,
              epochs=epochs,
              verbose=2,
              callbacks=[history] + callbacks)

    # evaluate the model
    logger.info("Evaluate...")
    return _listify(model.evaluate(valid[0], valid[1])), _format_keras_history(history)

def take_first_asis(x):
    """Take first argument as is
    """
    if hasattr(x, "__len__"):
        return x[0]
    else:
        return x


class CompileFN():

    def __init__(self, db_name, exp_name,  # TODO - check if we can somehow get those from hyperopt
                 data_fn,
                 model_fn,
                 eval2loss_fn=take_first_asis,
                 # validation
                 valid_split=.2,
                 cv_n_folds=None,
                 stratified=False,
                 random_state=None,
                 # saving
                 save_dir=DEFAULT_SAVE_DIR,
                 save_model=True,
                 save_results=True,
                 ):
        self.data_fn = data_fn
        self.model_fn = model_fn
        self.loss_fn = eval2loss_fn

        self.data_name = data_fn.__code__.co_name
        self.model_name = model_fn.__code__.co_name
        self.loss_name = eval2loss_fn.__code__.co_name
        self.db_name = db_name
        self.exp_name = exp_name
        # validation
        self.valid_split = valid_split
        self.cv_n_folds = cv_n_folds
        self.stratified = stratified
        self.random_state = random_state
        # saving
        self.save_dir = save_dir
        self.save_model = save_model
        self.save_results = save_results

    def __call__(self, param):
        time_start = datetime.now()

        # set default early-stop parameters
        if param.get("fit") is None:
            param["fit"] = {}
        if param["fit"].get("epochs") is None:
            param["fit"]["epochs"] = 500
        if param["fit"].get("patience") is None:
            param["fit"]["patience"] = 10
        if param["fit"].get("batch_size") is None:
            param["fit"]["batch_size"] = 32
        callbacks = [EarlyStopping(patience=param["fit"]["patience"])]

        # setup paths for storing the data - TODO check if we can somehow get the id from hyperopt
        rid = str(uuid4())
        tm_dir = self.save_dir + "/{db}/{exp}/train_models/".format(db=self.db_name, exp=self.exp_name)
        os.makedirs(tm_dir, exist_ok=True)
        model_path = tm_dir + "{0}.h5".format(rid) if self.save_model else ""
        results_path = tm_dir + "{0}.json".format(rid) if self.save_results else ""
        # -----------------

        # get data
        logger.info("Load data...")
        train, _ = self.data_fn(**merge_dicts(param["data"], param.get("shared", {})))
        time_data_loaded = datetime.now()

        # model parameters
        model_param = merge_dicts({"train_data": train}, param["model"], param.get("shared", {}))

        # train & evaluate the model
        if self.cv_n_folds is None:
            # no cross-validation
            model = self.model_fn(**model_param)
            train_idx, valid_idx = split_train_test_idx(train,
                                                        self.valid_split,
                                                        self.stratified,
                                                        self.random_state)
            eval_metrics, history = _train_and_eval_single(train=subset(train, train_idx),
                                                           valid=subset(train, valid_idx),
                                                           model=model,
                                                           epochs=param["fit"]["epochs"],
                                                           batch_size=param["fit"]["batch_size"],
                                                           callbacks=deepcopy(callbacks))
            if model_path:
                model.save(model_path)
        else:
            # cross-validation
            eval_metrics_list = []
            history = []
            for i, (train_idx, valid_idx) in enumerate(split_KFold_idx(train,
                                                                       self.cv_n_folds,
                                                                       self.stratified,
                                                                       self.random_state)):
                logger.info("Fold {0}/{1}".format(i + 1, self.cv_n_folds))
                model = self.model_fn(**model_param)
                eval_m, history_elem = _train_and_eval_single(train=subset(train, train_idx),
                                                              valid=subset(train, valid_idx),
                                                              model=model,
                                                              epochs=param["fit"]["epochs"],
                                                              batch_size=param["fit"]["batch_size"],
                                                              callbacks=deepcopy(callbacks))
                print("\n")
                eval_metrics_list.append(np.array(eval_m))
                history.append(history_elem)
                if model_path:
                    model.save(model_path.replace(".h5", "_fold_{0}.h5".format(i)))

            # summarize the metrics, take average
            eval_metrics = np.array(eval_metrics_list).mean(axis=0).tolist()

        loss = self.loss_fn(eval_metrics)
        time_end = datetime.now()

        ret = {"loss": loss,
               "status": STATUS_OK,
               "eval": {_listify(model.metrics_names)[i]: eval_metrics[i]
                        for i in range(len(eval_metrics))},
               # additional info
               "param": param,
               "path": {
                   "model": model_path,
                   "results": results_path,
               },
               "name": {
                   "data": self.data_name,
                   "model": self.model_name,
                   "loss": self.loss_name,
               },
               "history": history,
               # execution times
               "time": {
                   "start": str(time_start),
                   "end": str(time_end),
                   "duration": {
                       "total": (time_end - time_start).total_seconds(),  # in seconds
                       "dataload": (time_data_loaded - time_start).total_seconds(),
                       "training": (time_end - time_data_loaded).total_seconds(),
                   }}}

        # optionally save information to disk
        if results_path:
            write_json(ret, results_path)
        logger.info("Done!")
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
