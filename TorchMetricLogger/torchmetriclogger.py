from dataclasses import dataclass
from typing import Any
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

    def add_loss(self, group_name, metric):
        if group_name in self.metrics:
            self.metrics[group_name](metric)
        else:
            self.metrics[group_name] = metric.metric_class()
            self.metrics[group_name](metric)
    
    def __call__(self, **label_prediction):
        for group_name, metric in label_prediction.items():
            # first do a score over all classes
            self.add_metric(
                group_name,
                metric
            )   
            
            # then add a score for each individual class
            if metric.class_names != None:
                for class_name in metric.class_names:
                    self.add_metric(
                        group_name + "_" + class_name,
                        metric
                    )
    
    def on_batch_end(self):
        for metric_object in self.metrics.values():
            metric_object.reduce() 

        if self.log_function is not None:
            log_output = {name: metric.history[-1] for name, metric in self.metrics.items()}
            self.log_function(log_output)