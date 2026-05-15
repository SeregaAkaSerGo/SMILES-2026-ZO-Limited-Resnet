"""Initialization for the CIFAR100 classification head."""
import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    # Small Xavier initialization keeps initial logits near zero while still
    # breaking symmetry.  It is more stable for SPSA than Kaiming-scale logits.
    nn.init.xavier_uniform_(layer.weight, gain=0.25)
    nn.init.zeros_(layer.bias)
