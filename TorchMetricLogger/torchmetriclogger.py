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
    
@dataclass
class TmlLoss:
    loss: Any   


class TorchMetricLogger():
    def __init__(self):
        self.metrics = {}
        
    def add_metric(self, group_name, metric):
        if group_name in self.metrics:
            self.metrics[group_name](metric)
        else:
            self.metrics[group_name] = metric.metric_class()
            self.metrics[group_name](metric)
    
    def log(self, **label_prediction):
        for group_name, metric in label_prediction.items():
            if isinstance(metric, TmlLoss):
                self.add_loss(
                    group_name,
                    metric
                )
            
            elif isinstance(metric, TmlMetric):
                self.add_metric(
                    group_name,
                    metric
                )   
            else:
                raise Exception("Input Error", f"please pass the data in the Dataclass Loss or Metric, you passed {type(bench)}.")
    
    def on_batch_end(self):
        pass