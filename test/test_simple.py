from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMetric, TmlLoss
import torch

def binary_accuracy(label, prediction):
    prediction = (prediction > 0.5).double()
    tp = (label == prediction).double()
    return torch.mean(tp)


def test_bin_accuracy():
    """
    Simply check if the binary accuracy function works before we test TorchMetricsLogger
    """

    labels = torch.ones(100)
    predictions = torch.zeros(100)
    accuracy = binary_accuracy(labels, predictions)
    
    assert accuracy == 0

    labels = torch.ones(100)
    predictions = torch.ones(100)
    accuracy = binary_accuracy(labels, predictions)
    
    assert accuracy == 1

    labels = torch.ones(100)

    half_positive = torch.zeros(100)
    half_positive[50:] = 1

    assert binary_accuracy(labels, half_positive) == 0.5

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