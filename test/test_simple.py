from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMetric, TmlLoss, TmlMetricFunction
import torch

class TMLBinaryAccuracy(TmlMetricFunction):
    def __init__(self):
        super().__init__()
        
    def calculate(self, gold_labels, predictions):
        return (gold_labels > 0.5) == (predictions > 0.5)

binary_accuracy = TMLBinaryAccuracy()

def test_bin_accuracy():
    """
    Simply check if the binary accuracy function works before we test TorchMetricsLogger
    """

    labels = torch.ones(100)
    predictions = torch.zeros(100)
    weights = torch.ones(100)
    metrics_weights = binary_accuracy(labels, predictions, weights)
    assert metrics_weights[0]/metrics_weights[1] == 0

    labels = torch.ones(100)
    predictions = torch.ones(100) 
    metrics_weights = binary_accuracy(labels, predictions, weights)
    assert metrics_weights[0]/metrics_weights[1] == 1

    labels = torch.ones(100)

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    metrics_weights = binary_accuracy(labels, half_positive, weights)
    assert metrics_weights[0]/metrics_weights[1] == 0.5

def test_tml_bin_accuracy_batches():
    """
    Test TorchMetricsLogger with one entry per Batch
    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    
    tml = TML()

    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy)
    )

    tml.batch_end()

    predictions = torch.ones(100)
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy)
    )

    tml.batch_end()

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, half_positive, binary_accuracy)
    )

    tml.batch_end()
    
    assert tml.history["train_bin_accuracy"] == [0, 1, 0.5] 

def test_tml_bin_accuracy_epoch():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    
    tml = TML()

    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy)
    )

    predictions = torch.ones(100)
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy)
    )

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, half_positive, binary_accuracy)
    )

    tml.batch_end()
    
    assert tml.history["train_bin_accuracy"] == [0.5]


def test_tml_bin_accuracy_epoch_weights():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    weights = torch.zeros(100)
    
    tml = TML()

    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy, weights=weights)
    )

    predictions = torch.ones(100)
    weights = torch.ones(100)

    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy, weights=weights)
    )

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, half_positive, binary_accuracy, weights=half_positive)
    )

    tml.batch_end()
    
    assert tml.history["train_bin_accuracy"] == [1] 


def test_tml_bin_accuracy_batch_different_groups():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    """
    labels = torch.ones((100, 4))
    predictions = torch.zeros((100, 4))
    
    tml = TML()

    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy, ["zero", "one", "two", "three"])
    )

    predictions = torch.ones((100, 4))
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, predictions, binary_accuracy, ["zero", "one", "two", "three"])
    )

    half_positive = torch.zeros((100, 4))
    half_positive[50:] = 1
    tml(partial=True, 
        train_bin_accuracy=TmlMetric(labels, half_positive, binary_accuracy, ["zero", "one", "two", "three"])
    )

    tml.batch_end()

    # check that we have the correct names
    assert list(tml.history.keys()) == ['train_bin_accuracy', 'train_bin_accuracy_zero', 'train_bin_accuracy_one', 'train_bin_accuracy_two', 'train_bin_accuracy_three']

    assert tml.history["train_bin_accuracy"] == [0.5] 
    assert tml.history["train_bin_accuracy_zero"] == [0.5] 
    assert tml.history["train_bin_accuracy_one"] == [0.5] 
    assert tml.history["train_bin_accuracy_two"] == [0.5] 
    assert tml.history["train_bin_accuracy_three"] == [0.5] 