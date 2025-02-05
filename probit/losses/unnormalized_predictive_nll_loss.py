"""Unnormalized predictive NLL loss for multiclass classification."""

from functools import partial

from torch import nn

from probit.losses.exp_nll_loss import ExpNLLLoss
from probit.losses.normcdf_nll_loss import NormCDFNLLLoss
from probit.losses.sigmoid_nll_loss import SigmoidNLLLoss
from probit.utils.predictive import PREDICTIVE_DICT


class UnnormalizedPredictiveNLLLoss(nn.Module):
    """Unnormalized predictive NLL loss."""

    def __init__(self, predictive, approximate):
        super().__init__()

        if not predictive.startswith(("probit", "logit", "log")) or predictive.endswith(
            "mc"
        ):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]

        if predictive.startswith("probit"):
            self._predictive = partial(self._predictive, approximate=approximate)

        if predictive.startswith("probit"):
            self._loss = NormCDFNLLLoss(approximate=approximate)
        elif predictive.startswith("logit"):
            self._loss = SigmoidNLLLoss()
        else:  # predictive.startswith("log")
            self._loss = ExpNLLLoss()

    def forward(self, logits, targets):
        preds = self._predictive(*logits, return_logits=True)
        loss = self._loss(preds, targets)

        return loss
