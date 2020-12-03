import torch
import numpy as np

class TmlMetricFunction():
    def __init__(self):
        self.partial = []
        self.history = []
    
    def __call__(self, metric):
        if torch.is_tensor(metric.weights):
            metric.weights = metric.weights.detach().numpy()
            
        if metric.weights is None:
            metric.weights = np.ones(metric.gold_labels.shape)
        
        if torch.is_tensor(metric.gold_labels):
            metric.gold_labels = metric.gold_labels.detach().numpy()
            
        if torch.is_tensor(metric.predictions):
            metric.predictions = metric.predictions.detach().numpy()
        
        self.partial.append(self.calculate(metric))

        return self
    
    
    def reduce(self):
        self.partial = np.array(self.partial)
        
        # calculate the weighted mean
        score = self.reduction_function()
        
        # reset the partial, really important to erase the year 2020 ;)
        self.partial = []

        self.history.append(score)
        
        return score
    
    def calculate(self, metric):
        return [((metric.gold_labels > 0.5) == (metric.predictions > 0.5)), metric.weights]
    
    def reduction_function(self):
        return self.partial[:, 0].sum() / self.partial[:, 1].sum()

        
class TMLBinaryAccuracy(TmlMetricFunction):
    def __init__(self):
        super().__init__() 
    

# testing axes at the moment
class TMLDiceCoefficient(TmlMetricFunction):   
    def __call__(self, metric):
        assert metric.predictions is not None
        
        if metric.weights is None:
            metric.weights = np.ones(len(metric.gold_labels))
            
        super().__call__(metric)
        
    def calculate(self, metric):
        smooth = 1
        
        # somehow numpy only likes tuples here
        axis = tuple(np.arange(1, metric.gold_labels.ndim, dtype=np.int))
        
        intersection = np.sum(metric.gold_labels * metric.predictions, axis=axis)
        score = (2. * intersection + smooth) / (np.sum(metric.gold_labels, axis=axis) + np.sum(metric.predictions, axis=axis) + smooth)
        return [score, metric.weights]

    def reduction_function(self):
        score = np.average(self.partial[:, 0], weights=self.partial[:, 1]) 
        return score


# take the (weighted) mean, usefull for loss functions
class TMLMean(TmlMetricFunction):
    def __call__(self, metric):
        assert metric.gold_labels is None
        metric.predictions = np.array(metric.predictions)

        if metric.weights is None:
            metric.weights = np.ones(metric.predictions.size)
            
        super().__call__(metric)
        
    def calculate(self, metric):
        return [metric.predictions, metric.weights]

    def reduction_function(self):
        score = np.average(self.partial[:, 0], weights=self.partial[:, 1]) 
        return score