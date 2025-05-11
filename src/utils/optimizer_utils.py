"""
Methods to compute the total number of training steps and the step-wise learning rate scheduler.

The step-wise learning rate scheduler is based on the one proposed in the original GRPO paper:
https://arxiv.org/pdf/2402.03300
"""
import torch
from torch import nn
from typing import Tuple, Optional
from torch.optim.lr_scheduler import LambdaLR

def compute_total_steps(
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    len_dataset: int,
) -> int:
    """
    Compute the total number of training steps.
    """
    return int(num_train_epochs * (len_dataset / (per_device_train_batch_size * gradient_accumulation_steps)))


def get_step_wise_scheduler(
    optimizer,
    num_training_steps: int,
    warmup_steps: int = 2000,
    base_lr: float = 0.,
    peak_lr: float = 5.3e-4,
):
    """
    Step-wise learning rate scheduler following the one proposed in the original GRPO paper:
        Warmup phase: linear warmup from base_lr to peak_lr
        Decay phase: 
            - step-wise decay from peak_lr to 0.316 * peak_lr at 80% of the total training steps
            - step-wise decay to 0.1 * peak_lr at 90% of the total training steps
    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps (default: 2000)
        base_lr: Base learning rate (if None, uses optimizer's initial lr)
        peak_lr: Peak learning rate
    
    Returns:
        LambdaLR: A learning rate scheduler that can be used with Huggingface's Trainer
    """
    # Calculate step thresholds
    step_80 = int(0.8 * num_training_steps)
    step_90 = int(0.9 * num_training_steps)
    
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return (peak_lr - base_lr) * (current_step / warmup_steps)
        
        # After warmup, apply step-wise decay
        if current_step < step_80:
            return 1.0
        elif current_step < step_90:
            return 0.316
        else:
            return 0.1
    
    return LambdaLR(optimizer, lr_lambda)


def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    if forbidden_layer_names is None:
        forbidden_layer_names = []
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(forbidden in f"{name}.{n}".lower() for forbidden in forbidden_layer_names)
        ]
    # Add model specific parameters that are not in any child
    result += [
        k for k in model._parameters.keys() if not any(forbidden in k.lower() for forbidden in forbidden_layer_names)
    ]
    return result


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
    """
    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
    return decay_parameters


def get_grpo_optimizer(
    model,
    peak_learning_rate: float,
    adam_beta1: float,
    adam_beta2: float,
    weight_decay: float,
):

    decay_parameters = get_decay_parameter_names(model=model)
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=peak_learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=weight_decay,
    )


def get_grpo_scheduler(
    grpo_optimizer,
    total_steps,
    warmup_steps=2000,
    base_lr=0.,
    peak_lr=5.3e-4,
):

    return get_step_wise_scheduler(
        optimizer=grpo_optimizer,
        num_training_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        peak_lr=peak_lr,
    )