"""Train the models
"""
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
        train, _ = self.data_fun(**merge_dicts(param["data"], param.get("shared", {})))
        _test_len(train)  # check the data validity at this stage
        time_data_loaded = datetime.now()

        # compute the sequence length etc
        param = self.update_param_fun(param, train)

        # convenience wrapper
        def get_model(model_fun, param, x):
            model = model_fun(**merge_dicts(param["model"], param.get("shared", {})))
            # weightnorm, initialize on a batch of data
            if param["model"].get("use_weightnorm", False):
                # TODO - restrict only to a fraction of the training set
                data_based_init(model, x)
            return model

        # train & evaluate the model
        if self.cv_n_folds is None:
            # no cross-validation
            model = get_model(self.model_fun, param, train[0])
            train_idx, valid_idx = _train_test_split(train, self.valid_split, self.stratified, self.random_state)
            c_train, c_valid = _train_test_split_by_idx(train, train_idx, valid_idx)
            eval_metrics, history = _train_and_eval_single(train=c_train,
                                                           valid=c_valid,
                                                           model=model,
                                                           epochs=param["fit"]["epochs"],
                                                           callbacks=deepcopy(callbacks))
            if model_path:
                model.save(model_path)
        else:
            # cross-validation
            eval_metrics_list = []
            history = []
            for i, (train_idx, valid_idx) in enumerate(_get_kf(train, self.cv_n_folds, self.stratified, self.random_state)):
                logger.info("Fold {0}/{1}".format(i + 1, self.cv_n_folds))
                c_train, c_valid = _train_test_split_by_idx(train, train_idx, valid_idx)
                model = get_model(self.model_fun, param, train[0])
                eval_metrics, history_elem = _train_and_eval_single(train=c_train,
                                                                    valid=c_valid,
                                                                    model=model,
                                                                    epochs=param["fit"]["epochs"],
                                                                    callbacks=deepcopy(callbacks))
                print("\n")
                eval_metrics_list.append(np.array(eval_metrics))
                history.append(history_elem)
                if model_path:
                    model.save(model_path.replace(".h5", "_{0}.h5".format(i)))

            # summarize the metrics, take average
            eval_metrics = np.array(eval_metrics_list).mean(axis=0).tolist()

        loss = self.loss_fun(eval_metrics)
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


def _format_keras_history(history):
    """nicely format keras history
    """
    return {"params": history.params,
            "loss": merge_dicts({"epoch": history.epoch}, history.history),
            }


def _train_and_eval_single(train, valid, model, epochs=300, callbacks=[]):
    """Fit and evaluate a single model
    """
    # train the model
    logger.info("Fit...")
    history = History()
    model.fit(train[0], train[1],
              validation_data=valid,
              epochs=epochs,
              verbose=2,
              callbacks=[history] + callbacks)

    # evaluate the model
    logger.info("Evaluate...")
    return model.evaluate(valid[0], valid[1]), _format_keras_history(history)


def _train_test_split(train, valid_split=.2, stratified=False, random_state=None):
    y = train[1]
    x = np.arange(y.shape[0])
    stratify = y if stratified else None
    return train_test_split(x, test_size=valid_split, random_state=random_state, stratify=stratify)


def _get_kf(train, cv_n_folds=5, stratified=False, random_state=None):
    """Get k-fold indices generator
    """
    y = train[1]
    n_rows = y.shape[0]
    if stratified:
        if len(y.shape) > 1:
            if y.shape[1] > 1:
                raise ValueError("Can't use stratified K-fold with multi-column response variable")
            else:
                y = y[:, 0]
            # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold.split
        return StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=random_state)\
            .split(X=np.zeros((n_rows, 1)), y=y)
    else:
        return KFold(n_splits=cv_n_folds, shuffle=True, random_state=random_state)\
            .split(X=np.zeros((n_rows, 1)))


def _train_test_split_by_idx(train, train_idx, test_idx):
    """Take a training dataset and split it into train, valid set,
    given the (train, test) indicies
    """
    assert train[1].shape[0] == len(train_idx) + len(test_idx)

    # y split
    train_y = train[1][train_idx]
    test_y = train[1][test_idx]

    # x split
    if isinstance(train[0], (list, tuple)):
        train_x = [x[train_idx] for x in train[0]]
        test_x = [x[test_idx] for x in train[0]]
    elif isinstance(train[0], dict):
        train_x = {k: v[train_idx] for k, v in train[0].items()}
        test_x = {k: v[test_idx] for k, v in train[0].items()}
    elif isinstance(train[0], np.ndarray):
        train_x = train[0][train_idx]
        test_x = train[0][test_idx]
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

    return (train_x, train_y), (test_x, test_y)


def _test_len(train):
    """Test if all the elements in the training have the same shape[0]
    """
    l = train[1].shape[0]
    if isinstance(train[0], dict):
        for x in train[0].keys():
            assert train[0][x].shape[0] == l
    elif isinstance(train[0], (list, tuple)):
        for x in train[0]:
            assert x.shape[0] == l
    elif isinstance(train[0], np.ndarray):
        assert train[0].shape[0] == l
    else:
        raise ValueError("train[0] can only be of type: list, tuple, dict or np.ndarray")
