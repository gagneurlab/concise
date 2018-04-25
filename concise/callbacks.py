""" EvalMetric Callback: utility called at certain points during model training.
Author: Mohammadamin Barekatain
Affiliation: TUM
"""

import keras
import numpy as np
from concise.utils.helper import _to_string


class EvalMetric(keras.callbacks.Callback):

    """Evaluate the single-task or multi-task model after every epoch via the given metric.
    Evaluation is done for each individual task and the average is also reported.
    Evaluation results are added to logs so that they can be used with
    other callbacks such as TensorBoard, EarlyStopping, ModelCheckpoint
    when this callback is added to callback list earlier.
    # Arguments
        tasks: List of strings, task names which is used to retrieve the labels for each task.
        eval_metrics: A list or a dictionary of functions which are used for evaluation.
        In case a list or a simple function are provided, metric names are inferred from the function name.
        verbose: verbosity mode, 0 or 1. 1 gives a more verbose output.
        train_data: Training data used for evaluation. If None, the evaluation is done only on validation data.
    """

    def __init__(self, tasks, eval_metrics, verbose=1, train_data=None):
        self.tasks = tasks
        if isinstance(eval_metrics, dict):
            self.eval_metrics = eval_metrics
        else:
            self.eval_metrics = {_to_string(metric): metric
                                   for metric in eval_metrics}
        self.train_data = train_data
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.val_metric_log = {}
        self.train_metric_log = {}
        self.val_average_metric_log = {}
        self.train_average_metric_log = {}

        for eval_metric_name, eval_metric in self.eval_metrics.items():
            self.val_metric_log[eval_metric_name]={}
            self.train_metric_log[eval_metric_name]={}
            for task in self.tasks:
                self.params['metrics'].append(
                    'val_' + eval_metric_name + '_' + task)
                self.params['metrics'].append(
                    'train_' + eval_metric_name + '_' + task)
                self.val_metric_log[eval_metric_name].update({task:[]})
                self.train_metric_log[eval_metric_name].update({task:[]})

            self.val_average_metric_log[eval_metric_name] = []
            self.train_average_metric_log[eval_metric_name] = []
            self.params['metrics'].append('val_' + eval_metric_name + '_avg')
            self.params['metrics'].append('train_' + eval_metric_name + '_avg')

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        verbose = self.verbose
        self.num_input_fea = len(self.validation_data) - 2 * len(self.tasks) - 1
        # ------
        # evaluate for validation set
        y_pred_valid = self.model.predict([self.validation_data[i] for i in range(self.num_input_fea)])
        if len(y_pred_valid) != len(self.tasks):
            y_pred_valid = np.expand_dims(y_pred_valid, axis=0)

        for eval_metric_name, eval_metric in self.eval_metrics.items():
            val_values=[]
            for i, task in enumerate(self.tasks):
                value = eval_metric(self.validation_data[i+self.num_input_fea], y_pred_valid[i])
                self.val_metric_log[eval_metric_name][task].append(value)
                logs['val_' + eval_metric_name + '_' + task] = value
                val_values.append(value)
                if verbose > 0:
                    print(' - val_' + eval_metric_name + '_' + task + ':', value, end='')

            self.val_average_metric_log[eval_metric_name].append(np.average(val_values))
            logs['val_' + eval_metric_name + '_avg'] = np.average(val_values)
            if verbose > 0:
                print(' - val_' + eval_metric_name + '_avg:', np.average(val_values))

        # ------
        # evaluate for train set
        for eval_metric_name, eval_metric in self.eval_metrics.items():
            if self.train_data:
                y_pred_train = self.model.predict(self.train_data[0])
                if len(y_pred_train) != len(self.tasks):
                    y_pred_train = np.expand_dims(y_pred_train, axis=0)

                train_values=[]
                for i, task in enumerate(self.tasks):
                    value = eval_metric(self.train_data[1][task], y_pred_train[i])
                    self.train_metric_log[eval_metric_name][task].append(value)
                    logs['train_' + eval_metric_name + '_' + task] = value
                    train_values.append(value)
                    if verbose > 0:
                        print(' - train_' + eval_metric_name + '_' + task + ':', value, end='')

                self.train_average_metric_log[eval_metric_name].append(np.average(train_values))
                logs['train_' + eval_metric_name + '_avg'] = np.average(train_values)
                if verbose > 0:
                    print(' - train_' + eval_metric_name + '_avg:', np.average(train_values))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

