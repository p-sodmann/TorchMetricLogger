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

    def __post_init__(self):
        """This generates a partial dictionary for current runs, as well as a history object.
        """
        self.partial = defaultdict(list)
        self.history = defaultdict(list)

        # check if all needed parameters are given
        self.check_requirements()

    def make_numpy(self, metric):
        """Turn pytorch tensors into numpy arrays, if needed. This also deals with missing weights.

        Returns:
            [type]: [description]
        """
        if torch.is_tensor(metric.weights):
            metric.weights = metric.weights.detach().cpu().numpy()

        if torch.is_tensor(metric.gold_labels):
            metric.gold_labels = metric.gold_labels.detach().cpu().numpy()

        if torch.is_tensor(metric.values):
            metric.values = metric.values.detach().cpu().numpy()

        if torch.is_tensor(metric.predictions):
            metric.predictions = metric.predictions.detach().cpu().numpy()

        return metric

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
        metric = self.make_numpy(metric)
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


class TMLBinaryAccuracy(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        dims = self.dims(metric)

        tp = np.sum((metric.gold_labels > 0.5) * (metric.predictions > 0.5), axis=dims)
        tn = np.sum((metric.gold_labels < 0.5) * (metric.predictions < 0.5), axis=dims)
        fn = np.sum((metric.gold_labels > 0.5) * (metric.predictions < 0.5), axis=dims)
        fp = np.sum((metric.gold_labels < 0.5) * (metric.predictions > 0.5), axis=dims)

        return {
            # only count positives
            # correct for length of answers
            "metric": (tp + tn) / np.clip(tp + fp + tn + fn, 1, None),
            "weights": metric.weights,
        }


class TMLDice(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        dims = self.dims(metric)

        tp = np.sum((metric.gold_labels > 0.5) * (metric.predictions > 0.5), axis=dims)
        fp = np.sum((metric.gold_labels < 0.5) * (metric.predictions > 0.5), axis=dims)
        fn = np.sum((metric.gold_labels > 0.5) * (metric.predictions < 0.5), axis=dims)
        
        return {
            # only count positives
            # correct for length of answers
            "tps": tp,
            "fps": fp,
            "fns": fn,
            # prevent dice of 0 if mask is empty, also prevent division by zero
            "metric": np.clip(2*tp, 1, None) / np.clip(2*tp + fp + fn, 1, None),
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
            # median not weighted
            "micro": np.clip(2*tp, 1, None) / np.clip(2*tp + fp + fn, 1, None)
        }


class TMLF1(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):
        # in case this is one dim array
        dims = self.dims(metric)

        tp = (metric.gold_labels > 0.5) * (metric.predictions > 0.5)
        fp = (metric.gold_labels < 0.5) * (metric.predictions > 0.5)
        fn = (metric.gold_labels > 0.5) * (metric.predictions < 0.5)
        
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
        tp = np.array(self.partial["tps"])
        fp = np.array(self.partial["fps"])
        fn = np.array(self.partial["fns"])
        
        s_tp = np.array(self.partial["s_tps"])
        s_fp = np.array(self.partial["s_fps"])
        s_fn = np.array(self.partial["s_fns"])

        precision = tp.sum() / np.clip(tp.sum() + fp.sum(), a_min=1, a_max=None)
        recall = tp.sum() / np.clip(tp.sum() + fn.sum(), a_min=1, a_max=None)
        
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
            "precision": precision,
            "recall": recall,
            # median not weighted
            "micro": (2*tp.sum()) / np.clip(2*tp.sum() + fp.sum() + fn.sum(), 1, None),
            "soft_micro": (2*s_tp.sum()) / np.clip(2*s_tp.sum() + s_fp.sum() + s_fn.sum(), 1, None),
            "tp": tp.sum(),
            "fp": fp.sum(),
            "fn": fn.sum()
        }
