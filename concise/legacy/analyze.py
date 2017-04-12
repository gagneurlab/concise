# functions to analyze the results in python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concise.utils.helper import merge_dicts


# make a report
def get_cv_accuracy(res):
    """
    Extract the cv accuracy from the model
    """
    ac_list = [(accuracy["train_acc_final"],
                accuracy["test_acc_final"]
                )
               for accuracy, weights in res]
    ac = np.array(ac_list)

    perf = {
        "mean_train_acc": np.mean(ac[:, 0]),
        "std_train_acc": np.std(ac[:, 0]),
        "mean_test_acc": np.mean(ac[:, 1]),
        "std_test_acc": np.std(ac[:, 1]),

    }
    return perf


def get_kwargs_cv_accuracy(cv_res, i=None, filename=None):
    a = cv_res['kwargs']
    b = get_cv_accuracy(cv_res['output'])
    dic = merge_dicts(a, b)

    # append i if neccesary
    if i is not None:
        dic = merge_dicts(dic, {'i': i})
    if i is not None:
        dic = merge_dicts(dic, {'filename': filename})
    # append motifs, execution time and features:
    dic = merge_dicts(dic, {'features': cv_res.get('features', None)})
    dic = merge_dicts(dic, {'execution_time': cv_res.get('execution_time', None)})
    dic = merge_dicts(dic, {'motifs': cv_res.get('motifs', None)})
    return dic


# update this function
def cv_list2dt(cv_list):
    perf_list = [get_kwargs_cv_accuracy(res, i=i, filename=filename) for res, i, filename in cv_list]

    dt = pd.DataFrame(perf_list)
    return dt


def print_report(weights):
    np.set_printoptions(suppress=True, precision=2)
    print("motif tensor:")
    print(weights["motif_base_weights"])

    print("motif_weights:")
    print(weights["motif_weights"])
    print("motif_bias:")
    print(weights["motif_bias"].reshape([-1, 1]))
    print("final_bias")
    print(weights["final_bias"])

    print("feature_weights")
    print(weights["feature_weights"])

    def myfunc(x):
        return ['A', 'C', 'G', 'T'][x]

    # get the maximal index at each position:
    print("argmax letter:\n")
    best_letter = np.argmax(weights["motif_base_weights"], axis=1)
    vfunc = np.vectorize(myfunc)
    print(np.reshape(np.array(([''.join(i) for i in vfunc(best_letter)])), [-1, 1]))

    print("\nargmin letter:\n")
    worst_letter = np.argmin(weights["motif_base_weights"], axis=1)
    print(np.reshape(np.array(([''.join(i) for i in vfunc(worst_letter)])), [-1, 1]))


def plot_accuracy(accuracy):
    # plot the results
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(accuracy['loss_history'], 'o')
    plt.ylim(ymax=0.5)
    plt.xlabel('Iteration')
    plt.subplot(2, 1, 2)
    plt.title('Mean squared error')
    plt.plot(accuracy['step_history'], accuracy['train_acc_history'], '-o', label='train')
    plt.plot(accuracy['step_history'], accuracy['val_acc_history'], '-o', label='val')
    plt.ylim(ymin=0.05, ymax=0.3)
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()


def plot_pos_bias(weights):
    plt.plot(weights['spline_pred'])
    # plt.ylim(ymin=0, ymax=2)
    # prevent using offset
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    plt.show()
