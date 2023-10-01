from collections import defaultdict
import torch
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class TmlMetric:
    predictions: Any = None
    gold_labels: Any = None
    values: Any = None
    metric_class: Any = None
    class_names: Any = None
    weights: Any = None
    is_metric: bool = True
    log_function: Any = None

    def __post_init__(self):
        """This generates a partial dictionary for current runs, as well as a history object.
        """
        self.partial = defaultdict(list)
        self.history = defaultdict(list)

        # check if all needed parameters are given
        self.check_requirements()
        self(self)

    def make_numpy(self):
        """Turn pytorch tensors into numpy arrays, if needed. This also deals with missing weights.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(self.weights):
            self.weights = self.weights.float().detach().cpu().numpy()

        if torch.is_tensor(self.gold_labels):
            self.gold_labels = self.gold_labels.float().detach().cpu().numpy()

        if torch.is_tensor(self.values):
            self.values = self.values.float().detach().cpu().numpy()

        if torch.is_tensor(self.predictions):
            self.predictions = self.predictions.float().detach().cpu().numpy()

    def check_requirements(self):
        pass
    
    def dims(self, metric):
        if metric.gold_labels.ndim > 1:
            return tuple(np.arange(1, metric.gold_labels.ndim).tolist())
        else:
            return 0

    def reduce(self):
        # calculate the weighted mean
        scores = self.reduction_function()

        # reset the partial
        self.partial = defaultdict(list)
        for key, value in scores.items():
            self.history[key].append(value)

        return scores
    
    def __call__(self, metric):
        # make sure, we get the same type of metric everytime
        assert type(self) == type(metric)

        # make anything a numpy array and generate weights
        self.make_numpy()
        result = self.calculate(metric)

        for key, value in result.items():
            # this is ugly.
            if isinstance(value, Iterable):
                if value.ndim > 0:
                    self.partial[key].extend(value)
                else:
                    self.partial[key].append(value)
            elif value is not None:
                self.partial[key].append(value)

        return self

    def reduction_function(self):
        if "weights" in self.partial:
            metric_mean = np.average(
                self.partial["metric"], weights=self.partial["weights"]
            )
        else:
            metric_mean = np.mean(self.partial["metric"])
            
        return {
            "mean": metric_mean,
            # median not weighted
            #"median": np.median(self.partial["metric"]),
            #"min": float(np.min(self.partial["metric"])),
            #"max": float(np.max(self.partial["metric"])),
        }

class TmlPlot(TmlMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_metric = False
        self.called_log_function = kwargs.get("log_function", None)

class TmlMean(TmlMetric):
    def check_requirements(self):
        assert self.values is not None

    def calculate(self, metric):
        return {
            # only count positives
            # correct for length of answers
            "metric": metric.values,
            "weights": metric.weights,
        }


class TmlBinaryAccuracy(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        dims = self.dims(metric)

        tp = np.sum((metric.gold_labels >= 0.5) * (metric.predictions >= 0.5), axis=dims)
        tn = np.sum((metric.gold_labels < 0.5) * (metric.predictions < 0.5), axis=dims)
        fn = np.sum((metric.gold_labels >= 0.5) * (metric.predictions < 0.5), axis=dims)
        fp = np.sum((metric.gold_labels < 0.5) * (metric.predictions >= 0.5), axis=dims)

        return {
            # only count positives
            # correct for length of answers
            "metric": (tp + tn) / np.clip(tp + fp + tn + fn, 1, None),
            "weights": metric.weights,
        }

def calc_precision(tp, fp, fn):
    precision = tp.sum() / np.clip(tp.sum() + fp.sum(), a_min=1, a_max=None)
    return precision

def calc_recall(tp, fp, fn):
    recall = tp.sum() / np.clip(tp.sum() + fn.sum(), a_min=1, a_max=None)
    return recall
    
class TmlDice(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        dims = self.dims(metric)

        tp = np.sum((metric.gold_labels >= 0.5) * (metric.predictions >= 0.5), axis=dims)
        fp = np.sum((metric.gold_labels < 0.5) * (metric.predictions >= 0.5), axis=dims)
        fn = np.sum((metric.gold_labels >= 0.5) * (metric.predictions < 0.5), axis=dims)
        
        return {
            # only count positives
            # correct for length of answers
            "tps": tp,
            "fps": fp,
            "fns": fn,
            "metric": np.nan_to_num((2*tp) / (2*tp + fp + fn), nan=0.0),
            "weights": metric.weights,
        }

    def reduction_function(self):
        tp = np.sum(self.partial["tps"])
        fp = np.sum(self.partial["fps"])
        fn = np.sum(self.partial["fns"])

        if "weights" in self.partial:
            macro_dice = np.average(
                self.partial["metric"], weights=self.partial["weights"]
            )
        else:                           
            macro_dice = np.mean(self.partial["metric"])

        return {
            "macro": macro_dice,
            "precision": calc_precision(tp, fp, fn),
            "recall": calc_recall(tp, fp, fn),
            "tps": tp,
            "fps": fp,
            "fns": fn,
            # median not weighted
            "micro": np.nan_to_num((2*tp) / (2*tp + fp + fn), nan=0.0),
        }


class TmlF1(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        # in case this is one dim array
        dims = self.dims(metric)

        tp = (metric.gold_labels >= 0.5) * (metric.predictions >= 0.5)
        fp = (metric.gold_labels < 0.5) * (metric.predictions >= 0.5)
        fn = (metric.gold_labels >= 0.5) * (metric.predictions < 0.5)
        
        s_tp = (metric.gold_labels) * (metric.predictions)
        s_fp = (1 - metric.gold_labels) * (metric.predictions)
        s_fn = (metric.gold_labels) * (1 - metric.predictions)

        return {
            # only count positives
            # correct for length of answers
            "tps": tp,
            "fps": fp,
            "fns": fn,
            "s_tps": s_tp,
            "s_fps": s_fp,
            "s_fns": s_fn,
            #"metric": tp / np.clip(tp + (fp + fn) / 2, 1, None),
            "weights": metric.weights,
        }

    def reduction_function(self):
        try:
            tp = np.array(self.partial["tps"])
            fp = np.array(self.partial["fps"])
            fn = np.array(self.partial["fns"])

            s_tp = np.array(self.partial["s_tps"])
            s_fp = np.array(self.partial["s_fps"])
            s_fn = np.array(self.partial["s_fns"])

            s_precision = s_tp.sum() / np.clip(s_tp.sum() + s_fp.sum(), a_min=1, a_max=None)
            s_recall = s_tp.sum() / np.clip(s_tp.sum() + s_fn.sum(), a_min=1, a_max=None)

            if "weights" in self.partial:
                macro_dice = np.average(
                    (2*tp.sum(axis=0)) / np.clip(2*tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0), 1, None), weights=self.partial["weights"]
                )
            else:
                macro_dice = np.mean(
                    (2*tp.sum(axis=0)) / np.clip(2*tp.sum(axis=0) + fp.sum(axis=0) + fn.sum(axis=0), 1, None)
                )

            return {
                "macro": macro_dice,
                "precision": calc_precision(tp, fp, fn),
                "recall": calc_recall(tp, fp, fn),
                # median not weighted
                "micro": (2*tp.sum()) / np.clip(2*tp.sum() + fp.sum() + fn.sum(), 1, None),
                "soft_micro": (2*s_tp.sum()) / np.clip(2*s_tp.sum() + s_fp.sum() + s_fn.sum(), 1, None),
                "tp": tp.sum(),
                "fp": fp.sum(),
                "fn": fn.sum()
            }
        except:
            print("error while calculating ")
            return {
                "macro": 0,
                "precision": 0,
                "recall": 0,
                # median not weighted
                "micro": 0,
                "soft_micro": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0
            }


class TmlHistogram(TmlPlot):
    """
    this creates a small table, which is used for plotting.
    for each positive and negative label, create a list of predictions.
    this will be plotted as two different histograms, when the epoch finishes.
    """

    def calculate(self, metric):
        """
            get the predictions for each label, as well as the labels
        """
        return {
            "gold_labels": metric.gold_labels,
            "predictions": metric.predictions,
            "weights": metric.weights,
        }
    
    def reduction_function(self):
        """
        create a dictionary with {label_pos: [], label_neg: []}
        """
        print("reduce histogram")
        result_dict = {}

        preds = np.array(self.partial["predictions"])
        labels = np.array(self.partial["gold_labels"])

        for i, class_name in enumerate(self.class_names):
            result_dict[f"{class_name}_pos"] = [p for l, p in zip(labels[:, i], preds[:,i]) if l >= 0.5]
            result_dict[f"{class_name}_neg"] = [p for l, p in zip(labels[:, i], preds[:,i]) if l < 0.5]
        
        self.result_dict = result_dict

        return result_dict
    
    def metric_log_function(self):
        return self.called_log_function(self.result_dict)

        