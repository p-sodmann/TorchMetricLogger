from collections import defaultdict
import numpy as np


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
        
    def calc_metric(self, group_name, gold_label, prediction, class_names, metric_function, partial=False):
        """
        calculates the metric from self.metric_function and stores it in the history
        """
        # first add the mean total metric
        metric = float(metric_function(gold_label, prediction))
        self._add(group_name, metric, partial)
            
        # then for each class add its metric as well
        for index, class_name in enumerate(class_names):
            class_metric = float(metric_function(gold_label[:, index], prediction[:, index]))
        
            self._add(group_name + "_" + class_name, class_metric, partial)
            
    def add_loss(self, group_name, loss, partial=False):
        # in case this is a torch tensor, recast as float
        loss = float(loss)
        self._add(group_name, loss, partial)

    def __call__(self, partial=False, **label_prediction):
        """
        takes an arbitrary amount of different "keys" and a tuple with (label, prediction)
        if data is batched during training, call it with partial=True
        
        example metric(train=(10, 10), test=(9, 10))
        """
        for group_name, values in label_prediction.items():
            if len(values) == 1:
                loss = values[0]
                self.add_loss(group_name, loss, partial)
                
            elif len(values) == 4:
                gold_label = values[0]
                prediction = values[1]
                class_names = values[2]
                function_metric = values[3]
                
                self.calc_metric(
                    group_name, 
                    gold_label, 
                    prediction, 
                    class_names, 
                    function_metric, 
                    partial
                )
            
            else:
                raise Exception("Input Error", f"Metrics Class only accepts either 1 value or 4 values, you passed {len(values)}.")
            
    def batch_end(self):
        for group_name, metric in self.partial.items():           
            # calculate the mean per entry in the metric
            metric = np.mean(metric, axis=0)

            self.history[group_name].append(metric)
        
        self.partial = defaultdict(list)
        
        # log the metrics to w&b
        if self.log_function is not None:
            log_output = {name: metric[-1] for name, metric in self.history.items()}
            self.log_function(log_output)