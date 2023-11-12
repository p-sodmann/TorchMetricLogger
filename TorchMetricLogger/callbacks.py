import lightning as lit
import wandb
from TorchMetricLogger import TmlMean

class GradNormCallback(lit.pytorch.Callback):
    """
    Logs the gradient norm.
    """
    def __init__(self, tml):
        super().__init__()
        self.ema_grad = None
        self.tml = tml

    def on_before_optimizer_step(self, trainer, model, optimizer):

        gn = gradient_norm(model)
        if self.tml is None:
            wandb.log({
                "grad_norm": gn[0],
                "grad_norm_max": gn[1],
            })
        else:
            tml(
                grad_mean=TmlMean(values=gn[0]),
                grad_max=TmlMean(values=gn[1]),
            )
            
        if self.ema_grad is None:
            self.ema_grad = gn[0]
        else:
            self.ema_grad = self.ema_grad * 0.99 + gn[0] * 0.01

            if gn[1] > self.ema_grad * 2:
                print("Gradient explosion detected!")
                print(f"Max gradient: {gn[1]}, Name: {gn[2]}")

def gradient_norm(model):
    total_norm = 0.0
    largest_norm = 0.0
    largest_norm_name = ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
            if param_norm > largest_norm:
                largest_norm = param_norm
                largest_norm_name = name
    total_norm = total_norm ** (1. / 2)
    
    return total_norm, largest_norm, largest_norm_name