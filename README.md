# TorchMetricLogger
Small class to log metrics during training

```
from torchmetriclogger import TorchMetricLogger

# create a new log instance, you can also provide a log_function, 
# that receives a dictionary of metrics per epoch

metric_logger = TorchMetricLogger()

def binary_accuracy(label, prediction):
  prediction = torch.float(prediction > 0.5)
  return torch.sum(label == prediction)

criterion = torch.nn.BCELoss()

for epoch in range(100):
    ## TRAINING LOOP
    # init Loss
    model.train()
    
    # mini_batch loop
    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, mininterval=1):
        optimizer.zero_grad()
        
        output = model(sample["x"])
        
        loss = criterion(output_y, sample["y"])
                
        loss.backward()
        optimizer.step()
                
        metric_logger(partial=True, 
                    train_dice=(sample["y"], output_y, ["Class_one", "Class_two", "Class_three", "Class_four"], binary_accuracy), 
                    train_loss = ([loss])
                   )
    
    ## VALIDATION LOOP
    model.eval()
    valid_loss = 0.0
    
    with torch.no_grad():
        for sample in valid_loader:
            output_y = model.forward(sample["x"])
            
            loss = criterion(output_y, sample["y"])
            
            dice_metric(partial=True, 
                valid_dice=(sample["y"], output_y, ["Class_one", "Class_two", "Class_three", "Class_four"], binary_accuracy), 
                valid_loss = ([loss])
            )
            
    metric_logger.batch_end()
```
