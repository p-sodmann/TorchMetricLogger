from dataclasses import dataclass
from typing import Any

from TorchMetricLogger.torchmetricfunction import TMLMean
import torch
import numpy as np


@dataclass
class TmlMetric:
    gold_labels: Any
    predictions: Any
    metric_class: Any = None
    class_names: Any = None
    weights: Any = None

class TorchMetricLogger():
    def __init__(self, log_function = None):
        self.metrics = {}
        self.log_function = log_function
        
    def add_metric(self, group_name, metric):
        if group_name in self.metrics:
            self.metrics[group_name](metric)
        else:
            self.metrics[group_name] = metric.metric_class()
            self.metrics[group_name](metric)
    
    def __call__(self, **label_prediction):
        for group_name, metric in label_prediction.items():
            # first do a score over all classes

            original_weights = metric.weights

            self.add_metric(
                group_name,
                metric
            )   

            # then add a score for each individual class
            if metric.class_names != None:
                for index, class_name in enumerate(metric.class_names):

                    if metric.gold_labels is not None:
                        gold_labels = metric.gold_labels[:, index]
                    else:
                        gold_labels = None

                    if metric.predictions is not None:
                        predictions = metric.predictions[:, index]
                    else:
                        predictions = None
                    
                    if original_weights is not None:
                        if original_weights.ndim > 1:
                            weights = original_weights[:, index]
                            
                        else:
                            weights = original_weights
                            
                    else:
                        weights = None

                    sub_metric = TmlMetric(gold_labels, predictions, metric.metric_class, weights=weights)

                    self.add_metric(
                        group_name + "_" + class_name,
                        sub_metric
                    )
    
    def on_batch_end(self):
        for metric_object in self.metrics.values():
            metric_object.reduce() 

        if self.log_function is not None:
            log_output = {name: metric.history[-1] for name, metric in self.metrics.items()}
            self.log_function(log_output)