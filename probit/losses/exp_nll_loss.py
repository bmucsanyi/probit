"""Exp + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn


class ExpNLLLoss(nn.Module):
    """Exp + NLL loss."""

    def __init__(self):
        super().__init__()

        self.register_buffer("max", torch.tensor(1.0, dtype=torch.double) - 1e-15)

    def forward(self, logits, targets):
        targets = F.one_hot(targets, num_classes=logits.shape[-1])

        # Compute exp NLL loss
        act = logits.exp().clamp(max=1.0)

        # Compute loss
        loss = -torch.where(
            targets == 1,
            act.log(),
            torch.log1p(-act.double().clamp(max=self.max)).float(),
        )

        # Sum along the class dimension
        loss = loss.mean(dim=1)

        # Mean over the batch
        return loss.mean()
