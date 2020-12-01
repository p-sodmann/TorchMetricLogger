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