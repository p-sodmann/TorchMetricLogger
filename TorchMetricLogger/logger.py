from .metrics import TmlMetric

class TorchMetricLogger:
    def __init__(self, log_function=None):
        self.metrics = {}
        self.log_function = log_function
        self.epoch = 0

    def add_metric(self, group_name, metric):
        # if the metric is not present in our collection, initialize it.
        if group_name not in self.metrics:
            self.metrics[group_name] = metric
        else:
            self.metrics[group_name](metric)

    def __call__(self, **label_prediction):
        for group_name, metric in label_prediction.items():
            # first do a score over all classes

            original_weights = metric.weights

            self.add_metric(group_name, metric)

            # then add a score for each individual class
            if metric.class_names != None and metric.is_metric:
                for index, class_name in enumerate(metric.class_names):
                    gold_labels = None if metric.gold_labels is None else metric.gold_labels[:, index]
                    predictions = None if metric.predictions is None else metric.predictions[:, index]
                    values = None if metric.values is None else metric.values[:, index]

                    if original_weights is not None:
                        if original_weights.ndim > 1:
                            weights = original_weights[:, index]

                        else:
                            weights = original_weights

                    else:
                        weights = None

                    sub_metric = type(metric)(
                        predictions, gold_labels, values, metric.metric_class, weights=weights
                    )

                    self.add_metric(group_name + "_" + class_name, sub_metric)

    def on_batch_end(self):
        for metric_object in self.metrics.values():
            metric_object.reduce()

        log_output = {
            name: {key: values[-1] for key, values in metric.history.items()}
            for name, metric in self.metrics.items()
        }

        # flatten the log output
        log_output = {
            name + "_" + key: value for name, metric in log_output.items() for key, value in metric.items()
        }

        if self.log_function is not None:
            self.log_function(log_output)

        self.epoch += 1

        for metric in self.metrics.values():
            # assume its type is a plot FIXME
            if not metric.is_metric:
                if self.log_function is not None:
                    self.log_function(metric.metric_log_function())
                else:
                    print("No log function provided for metric: " + metric.name + ".")

        return log_output