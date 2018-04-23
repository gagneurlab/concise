""" EvalMetric Callback: utility called at certain points during model training.
Author: Mohammadamin Barekatain
Affiliation: TUM
"""

import keras
import numpy as np


class EvalMetric(keras.callbacks.Callback):

    """Evaluate the single-task or multi-task model after every epoch via the given metric.
    Evaluation is done for each individual task and the average is also reported.
    Evaluation results are added to logs so that they can be used with
    other callbacks such as TensorBoard, EarlyStopping, ModelCheckpoint
    when this callback is added to callback list earlier.
    # Arguments
        tasks: list of strings, task names which is used to retrieve the labels for each task.
        eval_metric: function(y_true, y_pred) which is used for evaluation.
        eval_metric_name: string, name of the evaluation metric. This is used when reporting the evaluations.
        num_input_fea: int, number of inputs that the model gets.
        verbose: verbosity mode, 0 or 1.
        train_data: If None, the evaluation is done only on validation data.
    """

    def __init__(self, tasks, eval_metric, eval_metric_name, num_input_fea, verbose=1, train_data=None):
        self.tasks = tasks
        self.eval_metric = eval_metric
        self.eval_metric_name = eval_metric_name
        self.train_data = train_data
        self.num_input_fea = num_input_fea
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.val_metric_log = {}
        self.train_metric_log = {}
        for task in self.tasks:
            self.params['metrics'].append(
                'val_' + self.eval_metric_name + '_' + task)
            self.params['metrics'].append(
                'train_' + self.eval_metric_name + '_' + task)
            self.val_metric_log[task] = []
            self.train_metric_log[task] = []

        self.val_average_metric_log = []
        self.train_average_metric_log = []
        self.params['metrics'].append('val_' + self.eval_metric_name + '_avg')
        self.params['metrics'].append(
            'train_' + self.eval_metric_name + '_avg')

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        verbose = self.verbose

        # ------
        # evaluate for validation set
        y_pred_valid = self.model.predict([self.validation_data[i] for i in range(self.num_input_fea)])
        if len(y_pred_valid) != len(self.tasks):
            y_pred_valid = np.expand_dims(y_pred_valid, axis=0)

        val_values=[]
        for i, task in enumerate(self.tasks):
            value = self.eval_metric(self.validation_data[i+self.num_input_fea], y_pred_valid[i])
            self.val_metric_log[task].append(value)
            logs['val_' + self.eval_metric_name + '_' + task] = value
            val_values.append(value)
            if verbose > 0:
                print (' - val_' + self.eval_metric_name + '_' + task + ':', value, end='')

        self.val_average_metric_log.append(np.average(val_values))
        logs['val_' + self.eval_metric_name + '_avg'] = np.average(val_values)
        if verbose > 0:
            print (' - val_' + self.eval_metric_name + '_avg:', np.average(val_values))

        # ------
        # evaluate for train set
        if self.train_data:
            y_pred_train = self.model.predict(self.train_data[0])
            if len(y_pred_train) != len(self.tasks):
                y_pred_train = np.expand_dims(y_pred_train, axis=0)

            train_values=[]
            for i, task in enumerate(self.tasks):
                value = self.eval_metric(self.train_data[1][task], y_pred_train[i])
                self.train_metric_log[task].append(value)
                logs['train_' + self.eval_metric_name + '_' + task] = value
                train_values.append(value)
                if verbose > 0:
                    print (' - train_' + self.eval_metric_name + '_' + task + ':', value, end='')

            self.train_average_metric_log.append(np.average(train_values))
            logs['train_' + self.eval_metric_name + '_avg'] = np.average(train_values)
            if verbose > 0:
                print(' - train_' + self.eval_metric_name + '_avg:', np.average(train_values))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

