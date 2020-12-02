class TmlMetricFunction():
    def __init__(self):
        pass
    
    def __call__(self, gold_labels, predicitions, weights):
        sum_weights = weights.sum()
        
        metrics = self.calculate(gold_labels, predicitions)
        sum_metrics = (metrics * weights).sum()

        return [float(sum_metrics), float(sum_weights)]
    
    def calculate(self, gold_labels, predictions):
        return (gold_labels > 0.5) == (predictions > 0.5)

# Example
class TMLBinaryAccuracy(TmlMetricFunction):
    def __init__(self):
        super().__init__()
        
    def calculate(self, gold_labels, predictions):
        return (gold_labels > 0.5) == (predictions > 0.5)

# testing axes at the moment
class TMLDiceCoefficient(TmlMetricFunction):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis=axis
        
    def calculate(self, gold_labels, predictions):
        smooth = 1
        
        intersection = torch.sum(gold_labels * predictions, axis=self.axis)
        score = (2. * intersection + smooth) / (torch.sum(gold_labels, axis=self.axis) + torch.sum(predictions, axis=self.axis) + smooth)
        return score