from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from typing import Any
import torch

@dataclass
class TmlMetric:
    gold_labels: Any
    predictions: Any
    function: Any = None
    class_names: Any = None
    weights: Any = None
    
@dataclass
class TmlLoss:
    loss: Any

@dataclass
class TmlMetricLog:
    metric: Any
    weight: Any

class TorchMetricLogger:
    def __init__(self, log_function=None):
        """
        log_function should accept a dictionary of values per epoch and log them somewhere
        like to tensorboard, weights&biases or neptune
        """
        
        # defaultdict, that stores our metrics sorted by a key
        self.history = defaultdict(list)
        
        # during a batch, we only get minibatch aka partial data
        # we calculate the metric at batch end
        self.partial = defaultdict(list)
        self.log_function = log_function
        
    def _add(self, group_name, metric, partial=False):
        if not partial:
            self.history[group_name].append(metric)
        else:
            self.partial[group_name].append(metric)
        
    def calc_metric(self, group_name, metric:TmlMetric, partial=False):
        """
        calculates the metric from self.metric_function and stores it in the history
        """
        assert metric.gold_labels.shape == metric.predictions.shape

        # if we dont set explicit weights, we set them to ones
        if metric.weights is None:
            metric.weights = torch.ones(metric.gold_labels.shape)

        metric_log = metric.function(metric.gold_labels, metric.predictions, metric.weights)

        self._add(group_name, metric_log, partial)
        
        if metric.class_names is not None:
            # then for each class add its metric as well
            for index, class_name in enumerate(metric.class_names):
                class_metric = metric.function(metric.gold_labels[:, index], metric.predictions[:, index], metric.weights[:, index])
        
                self._add(group_name + "_" + class_name, class_metric, partial)
            
    def add_loss(self, group_name, loss:TmlLoss, partial=False):
        # in case this is a torch tensor, recast as float
        loss = float(loss)
        self._add(group_name, loss, partial)

    def __call__(self, partial=False, **label_prediction):
        """
        takes an arbitrary amount of different "keys" and a tuple with (label, prediction)
        if data is batched during training, call it with partial=True
        
        example metric(train=(10, 10), test=(9, 10))
        """
        for group_name, bench in label_prediction.items():
            if isinstance(bench, TmlLoss):
                self.add_loss(group_name, bench.loss, partial)
            
            elif isinstance(bench, TmlMetric):
                self.calc_metric(
                    group_name, 
                    bench,
                    partial
                )   
            else:
                raise Exception("Input Error", f"please pass the data in the Dataclass Loss or Metric, you passed {type(bench)}.")
            
    def batch_end(self):
        for group_name, metric in self.partial.items():           
            # calculate the mean per entry in the metric
            metric_weight = np.mean(metric, axis=0)

            metric = metric_weight[0] / metric_weight[1]
            self.history[group_name].append(metric)
        
        self.partial = defaultdict(list)
        
        # log the metrics to w&b
        if self.log_function is not None:
            log_output = {name: metric[-1] for name, metric in self.history.items()}
            self.log_function(log_output)