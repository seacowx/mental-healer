import torch
from torch.optim.lr_scheduler import _LRScheduler


class StepWiseLRScheduler(_LRScheduler):


    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=2000,
        base_lr: float | None = None,
        peak_lr: float = 5.3e-4,
        last_epoch=-1,
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            total_steps: Total number of training steps
            warmup_steps: Number of warmup steps (default: 2000)
            peak_lr: Peak learning rate (if None, uses optimizer's initial lr)
            last_epoch: The index of last epoch (default: -1)
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr if base_lr else optimizer.param_groups[0]['lr']
        self.peak_lr = peak_lr
        
        # Calculate step thresholds
        self.step_80 = int(0.8 * total_steps)
        self.step_90 = int(0.9 * total_steps)
        
        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        current_step = self.last_epoch + 1
        
        # Warmup phase
        if current_step < self.warmup_steps:
            # Linear warmup from base_lr to peak_lr
            return [
                self.base_lr + (self.peak_lr - self.base_lr) * (current_step / self.warmup_steps) 
                for _ in self.base_lrs
            ]
        
        # After warmup, apply step-wise decay
        if current_step < self.step_80:
            # Maintain peak learning rate until 80% of training
            return [self.peak_lr for _ in self.base_lrs]
        elif current_step < self.step_90:
            # Decrease to 31.6% of peak after 80% of training
            return [self.peak_lr * 0.316 for _ in self.base_lrs]
        else:
            # Decrease to 10% of peak after 90% of training
            return [self.peak_lr * 0.1 for _ in self.base_lrs]


def get_grpo_optimizer_and_scheduler(
    model,
    total_steps,
    warmup_steps=2000,
    base_lr=1.0e-5,
    peak_lr=5.3e-4,
):