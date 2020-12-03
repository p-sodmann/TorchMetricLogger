from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TmlMetric, TmlMetricFunction, TMLBinaryAccuracy, TMLDiceCoefficient, TMLMean
import torch
import numpy as np

def test_bin_accuracy():
    """
    Simply check if the binary accuracy function works before we test TorchMetricsLogger
    """

    # check that the result is 1 if label == prediction
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    weights = torch.ones(100)

    metric = TmlMetric(labels, predictions, weights=weights)
    assert TMLBinaryAccuracy()(metric).reduce() == 0

    # check that the result is 0 if label != prediction
    labels = torch.ones(100)
    predictions = torch.ones(100) 
    
    metric = TmlMetric(labels, predictions, weights=weights)
    assert TMLBinaryAccuracy()(metric).reduce() == 1

    # check that the result is 0.5 if label == prediction in half the cases
    labels = torch.ones(100)
    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    
    metric = TmlMetric(labels, half_positive, weights=weights)
    assert TMLBinaryAccuracy()(metric).reduce() == 0.5

def test_tml_bin_accuracy_batches():
    """
    Test TorchMetricsLogger with one entry per Batch
    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    
    tml = TML()

    # add one batch with only mistakes
    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy)
    )

    tml.on_batch_end()

    # add one batch with only correct predictions
    predictions = torch.ones(100)
    
    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy)
    )

    tml.on_batch_end()

    # add one prediction with half correct
    half_positive = torch.zeros(100)
    half_positive[50:] = 1

    tml( 
        train_bin_accuracy=TmlMetric(labels, half_positive, TMLBinaryAccuracy)
    )

    tml.on_batch_end()
    
    assert tml.metrics["train_bin_accuracy"].history == [0, 1, 0.5] 

def test_tml_bin_accuracy_epoch():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    
    tml = TML()

    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy)
    )

    predictions = torch.ones(100)
    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy)
    )

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    tml(
        train_bin_accuracy=TmlMetric(labels, half_positive, TMLBinaryAccuracy)
    )

    tml.on_batch_end()
    
    assert tml.metrics["train_bin_accuracy"].history == [0.5]


def test_tml_bin_accuracy_epoch_weights():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    since we set the weight for the first and last batch to zero, the result should be 1

    """
    labels = torch.ones(100)
    predictions = torch.zeros(100)
    weights = torch.zeros(100)
    
    tml = TML()

    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy, weights=weights)
        )

    predictions = torch.ones(100)
    weights = torch.ones(100)

    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy, weights=weights)
    )

    half_positive = torch.zeros(100)
    half_positive[50:] = 1
    tml(
        train_bin_accuracy=TmlMetric(labels, half_positive, TMLBinaryAccuracy, weights=half_positive)
    )

    tml.on_batch_end()
    
    assert tml.metrics["train_bin_accuracy"].history  == [1] 


def test_tml_bin_accuracy_batch_different_groups():
    """
    We now simulate one epoch with 3 batches.
    The batches should get averaged mean([0,1,0.5]) == 0.5
    """
    labels = torch.ones((100, 4))
    predictions = torch.zeros((100, 4))
    
    tml = TML()

    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy, ["zero", "one", "two", "three"])
    )

    predictions = torch.ones((100, 4))
    tml(
        train_bin_accuracy=TmlMetric(labels, predictions, TMLBinaryAccuracy, ["zero", "one", "two", "three"])
    )

    half_positive = torch.zeros((100, 4))
    half_positive[50:] = 1
    tml(
        train_bin_accuracy=TmlMetric(labels, half_positive, TMLBinaryAccuracy, ["zero", "one", "two", "three"])
    )

    tml.on_batch_end()
    
    # check that we have the correct names
    assert list(tml.metrics.keys()) == ['train_bin_accuracy', 'train_bin_accuracy_zero', 'train_bin_accuracy_one', 'train_bin_accuracy_two', 'train_bin_accuracy_three']

    assert tml.metrics["train_bin_accuracy"].history == [0.5] 
    assert tml.metrics["train_bin_accuracy_zero"].history == [0.5] 
    assert tml.metrics["train_bin_accuracy_one"].history == [0.5] 
    assert tml.metrics["train_bin_accuracy_two"].history == [0.5] 
    assert tml.metrics["train_bin_accuracy_three"].history == [0.5] 

def test_loss():
    # create a list of fake losses between 2 and 1
    loss = np.linspace(2.0, 0.0, num=50)

    tml = TML()

    # add them together as one batch
    for l in loss:
        tml(loss=TmlMetric(None, l, TMLMean))

    tml.on_batch_end()

    # assert the average loss is 1
    assert np.allclose(tml.metrics["loss"].history, 1)