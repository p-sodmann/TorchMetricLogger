# TorchMetricLogger
Small class to log metrics during training

``` python
from TorchMetricLogger import TorchMetricLogger as TML
from TorchMetricLogger import TMLMean, TMLDice, TMLF1

# create a new log instance, you can also provide a log_function, 
# that receives a dictionary of metrics per epoch

metric_logger = TML()

def binary_accuracy(label, prediction):
  prediction = torch.float(prediction > 0.5)
  return torch.sum(label == prediction)

criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(100):
    ## TRAINING LOOP
    # init Loss
    model.train()
    
    # mini_batch loop
    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, mininterval=1):
        optimizer.zero_grad()
        
        output_y = model(sample["x"])
        
        loss = criterion(output_y, sample["y"])
                
        loss.backward()
        optimizer.step()
                
        metric_logger( 
            train_path_accuracy=TMLBinaryAccuracy(
                output_y.sigmoid(),
                p,
                class_names=self.class_names
            ),
            train_loss=TmlMean(
                values=loss
            )
        )
    
    ## VALIDATION LOOP
    model.eval()
    valid_loss = 0.0
    
    with torch.no_grad():
        for sample in valid_loader:
            output_y = model.forward(sample["x"])
            
            loss = criterion(output_y, sample["y"])
            
            metric_logger( 
                valid_path_accuracy=TMLBinaryAccuracy(
                    output_y.sigmoid(),
                    p,
                    class_names=self.class_names
                ),
                valid_loss=TmlMean(
                    values=loss
                )
            )
            
    metric_logger.batch_end()
```
