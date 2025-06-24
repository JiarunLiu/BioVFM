import os
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, roc_curve

from medmnist.info import INFO, DEFAULT_ROOT
from medmnist.evaluator import Evaluator

from tqdm import tqdm
from functools import partial
from sklearn.utils import resample 




""" Evaluation Metric Functions """

def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def get_metrics_based_on_conf_mat(y_true, y_score, task, threshold=0.5):
    """
    Compute metrics based on confusion matrix, so we don't have to calculate the metrics one by one
    """

    if task == "multi-label, binary-class":
        if type(threshold) == float:
            y_pre = y_score > threshold
        elif type(threshold) == list:
            y_pre = y_score > np.array(threshold).reshape(1, -1)
        else:
            raise ValueError("threshold should be a float or a list of floats")

        accuracy = 0
        balanced_accuracy = 0
        mcc = 0
        for label in range(y_true.shape[1]):
            accuracy += accuracy_score(y_true[:, label], y_pre[:, label])
            balanced_accuracy += balanced_accuracy_score(y_true[:, label], y_pre[:, label])
            mcc += matthews_corrcoef(y_true[:, label], y_pre[:, label])
        recall_micro = recall_score(y_true, y_pre, average='micro')
        recall_macro = recall_score(y_true, y_pre, average='macro')
        f1_micro = f1_score(y_true, y_pre, average='micro')
        f1_macro = f1_score(y_true, y_pre, average='macro')
        ret = {
            "accuracy": accuracy / y_true.shape[1],
            "recall_micro": recall_micro,
            "recall_macro": recall_macro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "balanced_accuracy": balanced_accuracy / y_true.shape[1],
            "mcc": mcc / y_true.shape[1]
        }
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        # y_pre = y_score > threshold
        if type(threshold) == float:
            y_pre = y_score > threshold
        elif type(threshold) == list:
            # y_pre = y_score > np.array(threshold).reshape(1, -1)
            y_pre = y_score > np.array(threshold).reshape(1, -1)
        else:
            raise ValueError("threshold should be a float or a list of floats")
        recall = recall_score(y_true, y_pre)
        f1 = f1_score(y_true, y_pre)
        ret = {
            "accuracy": accuracy_score(y_true, y_pre),
            "recall_micro": recall,
            "recall_macro": recall,
            "f1_micro": f1,
            "f1_macro": f1,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pre),
            "mcc": matthews_corrcoef(y_true, y_pre)
        }
    else:
        ret = {
            "accuracy": accuracy_score(y_true, np.argmax(y_score, axis=-1)),
            "recall_micro": recall_score(y_true, np.argmax(y_score, axis=-1), average='micro'),
            "recall_macro": recall_score(y_true, np.argmax(y_score, axis=-1), average='macro'),
            "f1_micro": f1_score(y_true, np.argmax(y_score, axis=-1), average='micro'),
            "f1_macro": f1_score(y_true, np.argmax(y_score, axis=-1), average='macro'),
            "balanced_accuracy": balanced_accuracy_score(y_true, np.argmax(y_score, axis=-1)),
            "mcc": matthews_corrcoef(y_true, np.argmax(y_score, axis=-1))
        }

    return ret


def compute_all_metrics(y_true, y_score, task, threshold=0.5):
    metrics = get_metrics_based_on_conf_mat(y_true, y_score, task, threshold)
    metrics["auc"] = getAUC(y_true, y_score, task)
    return metrics


def get_best_p_vals(pred, groundtruth, label_names, metric_func=matthews_corrcoef, verbose: bool = False):
    best_p_vals = dict()
    for idx, label_name in enumerate(label_names):
        y_true = groundtruth[:, idx]
        fpr, tpr, probabilities = roc_curve(y_true, pred[:, idx])  # get possible thresholds
        probabilities = probabilities[1:]
        probabilities.sort()

        # if len(probabilities) > 1000, then we need to sample some thresholds
        if len(probabilities) > 1000:
            print("Too many thresholds, sample some of them.")
            probabilities = probabilities[::int(len(probabilities) / 1000)]  # sample 1000 thresholds
            

        metrics_list = []
        for p in probabilities:
            y_pred = np.where(pred[:, idx] < p, 0, 1)
            metric = metric_func(y_true, y_pred)
            metrics_list.append(metric)

        best_index = np.argmax(metrics_list)
        best_p = probabilities[best_index]
        best_metric = metrics_list[best_index]
        if verbose:
            print("Best metric for {} is {}. threshold = {}.".format(label_name, best_metric, best_p))

        best_p_vals[label_name] = best_p
    return best_p_vals


''' Bootstrap and Confidence Intervals '''
def compute_cis(data, confidence_level=0.05):
    """
    FUNCTION: compute_cis
    ------------------------------------------------------
    Given a Pandas dataframe of (n, labels), return another
    Pandas dataframe that is (3, labels). 
    
    Each row is lower bound, mean, upper bound of a confidence 
    interval with `confidence`. 
    
    Args: 
        * data - Pandas Dataframe, of shape (num_bootstrap_samples, num_labels)
        * confidence_level (optional) - confidence level of interval
        
    Returns: 
        * Pandas Dataframe, of shape (3, labels), representing mean, lower, upper
    """
    data_columns = list(data)
    intervals = []
    for i in data_columns: 
        series = data[i]
        sorted_perfs = series.sort_values()
        lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
        upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
        lower = sorted_perfs.iloc[lower_index].round(4)
        upper = sorted_perfs.iloc[upper_index].round(4)
        mean = round(sorted_perfs.mean(), 4)
        interval = pd.DataFrame({i : [mean, lower, upper]})
        intervals.append(interval)
    intervals_df = pd.concat(intervals, axis=1)
    intervals_df.index = ['mean', 'lower', 'upper']
    return intervals_df


def bootstrap(y_pred, y_true, eval_func, n_samples=1000): 
    '''
    This function will randomly sample with replacement 
    from y_pred and y_true then evaluate `n` times
    and obtain AUROC scores for each. 
    
    You can specify the number of samples that should be
    used with the `n_samples` parameter. 
    
    Confidence intervals will be generated from each 
    of the samples. 
    
    Note: 
    * n_total_labels >= n_cxr_labels
        `n_total_labels` is greater iff alternative labels are being tested

    Args:
        * y_pred - np.array, shape (n_samples, n_total_labels)
        * y_true - np.array, shape (n_samples, n_cxr_labels)
        * eval_func - function, evaluation function to be used
        * n_samples - int, number of samples to use
    '''
    np.random.seed(97)
    y_pred # (500, n_total_labels)
    y_true # (500, n_cxr_labels) 
    
    idx = np.arange(len(y_true))
    
    boot_stats = []
    for i in tqdm(range(n_samples)): 
        sample = resample(idx, replace=True, random_state=i)
        y_pred_sample = y_pred[sample]
        y_true_sample = y_true[sample]
        
        sample_stats = eval_func(y_true_sample, y_pred_sample)
        boot_stats.append(sample_stats)

    boot_stats = pd.DataFrame(boot_stats) # pandas array of evaluations for each sample
    return boot_stats, compute_cis(boot_stats)


class MyEvaluatorBootstrap:

    def __init__(self, flag, split, size=None, root=DEFAULT_ROOT):
        self.flag = flag
        self.split = split

        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            self.size = size
            self.size_flag = f"_{size}"

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

        self.info = INFO[self.flag]

        if self.split in ["train", "val", "test"]:
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError
        
        self.compute_ci = True

        
    def evaluate(self, y_score, val_predictions=None, val_targets=None):
        assert y_score.shape[0] == self.labels.shape[0]

        task = self.info["task"]
        label_names = list(self.info["label"].values())

        # only for binary-class and multi-label, binary-class need to find the best threshold
        if val_predictions is not None and val_targets is not None:
            if type(val_targets) != np.ndarray:
                val_targets = val_targets.numpy()
            if type(val_predictions) != np.ndarray:
                val_predictions = val_predictions.numpy()

            if task == "multi-label, binary-class":
                best_p_vals = get_best_p_vals(val_predictions, val_targets, label_names, metric_func=matthews_corrcoef)
                best_p_vals = list(best_p_vals.values())
            elif task == 'binary-class':
                val_targets_onehot = np.eye(2, dtype=float)[val_targets.squeeze()]
                best_p_vals = get_best_p_vals(val_predictions, val_targets_onehot, label_names, metric_func=matthews_corrcoef)
                best_p_vals = float(list(best_p_vals.values())[1])
            else:
                best_p_vals = 0.5
        else:
            best_p_vals = 0.5

        # compute val_metrics
        if val_predictions is not None and val_targets is not None:
            val_metrics = compute_all_metrics(val_targets, val_predictions, task, threshold=best_p_vals)
            val_metrics = pd.DataFrame([val_metrics])
        else:
            val_metrics = None

        if self.compute_ci:
            eval_func = partial(compute_all_metrics, task=task, threshold=best_p_vals)
            boot_stats, cis = bootstrap(y_score, self.labels, eval_func)

        if type(best_p_vals) == float:
            best_p_vals_df = pd.DataFrame([[best_p_vals for i in range(len(label_names))]], columns=label_names)
        else:
            best_p_vals_df = pd.DataFrame([best_p_vals], columns=label_names)

        return boot_stats, cis, val_metrics, best_p_vals_df


if __name__ == "__main__":

    # # Test the bootstrap function
    # y_pred = np.load("output/debug/predictions.npy")
    # y_true = np.load("output/debug/labels.npy")
    # # boot_stats, cis = bootstrap(y_pred, y_true, compute_all_metrics)

    # evaluator = MyEvaluatorBootstrap("chestmnist", "test")
    # boot_stats, cis = evaluator.evaluate(y_pred, val_predictions=y_pred, val_targets=y_true)


    y_pred = np.load("output/debug_breast/predictions.npy")
    y_true = np.load("output/debug_breast/labels.npy")
    # boot_stats, cis = bootstrap(y_pred, y_true, compute_all_metrics)

    evaluator = MyEvaluatorBootstrap("breastmnist", "test")
    boot_stats, cis = evaluator.evaluate(y_pred, val_predictions=y_pred, val_targets=y_true)

    # medmnist_evaluator = Evaluator("chestmnist", "test")
    # med_score = medmnist_evaluator.evaluate(y_pred)
    # print(med_score)

    print(boot_stats)
    print(cis)
    print("Done!")