"""Exp + NLL loss for multiclass classification."""

from torch import nn


class RegularizedSoftmaxNLLLoss(nn.Module):
    """Regularized softmax + NLL loss encouraging sum(exp(logits)) approx. c."""

    def __init__(self, regularization_factor, target_normalization_value):
        super().__init__()
        self._regularization_factor = regularization_factor
        self._target_normalization_value = target_normalization_value
        self._loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        loss = self._loss(logits, targets)
        regularizer = (
            logits.exp()
            .sum(dim=-1)
            .sub(self._target_normalization_value)
            .square()
            .mean()
        )

        return loss + self._regularization_factor * regularizer
