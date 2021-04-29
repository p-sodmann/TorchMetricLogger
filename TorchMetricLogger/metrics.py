from collections import defaultdict
import torch
import numpy as np
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class TmlMetric:
    gold_labels: Any = None
    predictions: Any = None
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

        if metric.weights is None:
            if metric.gold_labels is not None:
                metric.weights = np.ones(metric.gold_labels.shape)
            elif metric.predictions is not None:
                metric.weights = np.ones(metric.predictions.shape)
            elif metric.values is not None:
                metric.weights = np.ones(metric.values.shape)
            else:
                raise Exception("need Gold Label, Prediciton or Value to do anything.")

        if torch.is_tensor(metric.gold_labels):
            metric.gold_labels = metric.gold_labels.detach().cpu().numpy()
        
        if torch.is_tensor(metric.values):
            metric.values = metric.values.detach().cpu().numpy()

        if torch.is_tensor(self.predictions):
            metric.predictions = metric.predictions.detach().cpu().numpy()

        return metric

    def check_requirements(self):
        pass

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

        # make anything a numpy array and 
        metric = self.make_numpy(metric)
        result = self.calculate(metric)

        for key, value in result.items():
            if isinstance(value, Iterable):
                self.partial[key].extend(value)
            else:
                self.partial[key].append(value)

        return self


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

    def reduction_function(self):
        score = {
            "mean": np.average(self.partial["metric"], weights=self.partial["weights"]),
            # median not weighted
            "median": np.median(self.partial["metric"]),
            "min": np.min(self.partial["metric"]),
            "max": np.max(self.partial["metric"]),
        }

        return score

class TMLBinaryAccuracy(TmlMetric):
    def check_requirements(self):
        assert self.gold_labels is not None
        assert self.predictions is not None

    def calculate(self, metric):

        tp = (metric.gold_labels > 0.5) * (metric.predictions > 0.5)
        tn = (metric.gold_labels < 0.5) * (metric.predictions < 0.5)

        return {
            # only count positives
            # correct for length of answers
            "metric": (tp + tn),
            "weights": metric.weights,
        }

    def reduction_function(self):
        score = {
            "mean": np.average(self.partial["metric"], weights=self.partial["weights"]),
            # median not weighted
            "median": np.median(self.partial["metric"]),
            "min": np.min(self.partial["metric"]),
            "max": np.max(self.partial["metric"]),
        }

        return score