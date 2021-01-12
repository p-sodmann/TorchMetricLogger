import torch
import numpy as np
from collections import defaultdict
from collections.abc import Iterable

class TmlMetricFunction():
    def __init__(self):
        self.partial = defaultdict(list)
        self.history = [] # defaultdict(list)

    def make_numpy(self, metric):
        if torch.is_tensor(metric.weights):
            metric.weights = metric.weights.detach().cpu().numpy()
        
        if torch.is_tensor(metric.gold_labels):
            metric.gold_labels = metric.gold_labels.detach().cpu().numpy()
            
        if torch.is_tensor(metric.predictions):
            metric.predictions = metric.predictions.detach().cpu().numpy()
        
        return metric
    
    def __call__(self, metric):
        metric = self.make_numpy(metric)
        
        if metric.weights is None:
            metric.weights = np.ones(metric.gold_labels.shape)
        
        result = self.calculate(metric)

        for key, value in result.items():
            if isinstance(value, Iterable):
                self.partial[key].extend(value)
            else:
                self.partial[key].append(value)

        return self
    
    
    def reduce(self):     
        # calculate the weighted mean
        scores = self.reduction_function()
        
        # reset the partial
        self.partial = defaultdict(list)
        self.history.append(scores)

        return scores
    
    def calculate(self, metric):
        return {
            "metric": ((metric.gold_labels > 0.5) == (metric.predictions > 0.5)), 
            "weights":metric.weights
        }
    
    def reduction_function(self):
        return np.sum(self.partial["metric"]) / np.sum(self.partial["weights"])

        
class TMLBinaryAccuracy(TmlMetricFunction):
    def __init__(self):
        super().__init__() 
    

# testing axes at the moment
class TMLDiceCoefficient(TmlMetricFunction):   
    def __call__(self, metric):
        assert metric.predictions is not None
        
        metric = self.make_numpy(metric)

        if metric.weights is None:
            metric.weights = np.ones(shape=metric.gold_labels.shape[0])

        super().__call__(metric)
        
    def calculate(self, metric):
        smooth = 1
        
        # somehow numpy only likes tuples here
        axis = tuple(np.arange(1, metric.gold_labels.ndim, dtype=np.int))
        
        intersection = np.sum(metric.gold_labels * metric.predictions, axis=axis)
        dice_coefficient = (2. * intersection + smooth) / (np.sum(metric.gold_labels, axis=axis) + np.sum(metric.predictions, axis=axis) + smooth)
        
        return {
            "metric": dice_coefficient, 
            "weights": metric.weights
        }

    def reduction_function(self):
        score = np.average(self.partial["metric"], weights=self.partial["weights"]) 
        return score


# take the (weighted) mean, usefull for loss functions
class TMLMean(TmlMetricFunction):
    def __call__(self, metric):
        assert metric.gold_labels is None
        if metric.weights is None:
            metric.weights = np.ones(1)

        super().__call__(metric)

        
    def calculate(self, metric):
        log = {
            "metric": np.reshape(metric.predictions, [-1]),
            "weights": metric.weights
        }

        return log

    def reduction_function(self):
        score = np.average(self.partial["metric"], weights=self.partial["weights"])
        return score

class TMLF1(TmlMetricFunction):
    def __call__(self, metric):
        if metric.weights is None:
            metric.weights = np.ones(metric.predictions.shape)
    
        super().__call__(metric)
        
    def calculate(self, metric):
        tp = np.sum(((metric.gold_labels >= 0.5) * (metric.predictions >= 0.5)) * metric.weights)
        fp = np.sum(((metric.gold_labels < 0.5) * (metric.predictions >= 0.5)) * metric.weights)
        fn = np.sum(((metric.gold_labels >= 0.5) * (metric.predictions < 0.5)) * metric.weights)
        
        return {
            "tp": tp, 
            "fp": fp, 
            "fn": fn
            }

    def reduction_function(self):
        tp = np.sum(self.partial["tp"], axis=0)
        fp = np.sum(self.partial["fp"], axis=0)
        fn = np.sum(self.partial["fn"], axis=0)

        return np.mean(tp / (tp + (fp + fn)/2 + 1e-12) )