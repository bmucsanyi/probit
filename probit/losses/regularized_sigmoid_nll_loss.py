"""Sigmoid + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn


class RegularizedSigmoidNLLLoss(nn.Module):
    """Regularized sigmoid + NLL loss encouraging sum(sigmoid(logits)) approx. c."""

    def __init__(self, regularization_factor, target_normalization_value):
        super().__init__()
        self._regularization_factor = regularization_factor
        self._target_normalization_value = target_normalization_value
        self._loss = nn.NLLLoss()

    def forward(self, logits, targets):
        log_sigmoids = F.logsigmoid(logits)
        log_probs = log_sigmoids - torch.logsumexp(log_sigmoids, dim=1).unsqueeze(1)

        loss = self._loss(log_probs, targets)
        regularizer = (
            logits.sigmoid()
            .sum(dim=-1)
            .sub(self._target_normalization_value)
            .square()
            .mean()
        )

        return loss + self._regularization_factor * regularizer
