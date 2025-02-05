"""Backpack extension for NormedNdtrNLLLoss."""

import torch
import torch.nn.functional as F
from backpack.core.derivatives.nll_base import NLLLossDerivatives
from backpack.extensions.secondorder.hbp.losses import HBPLoss
from torch import Tensor
from torch.distributions import Categorical

from probit.utils.predictive import log_ndtr, log_normed_ndtr


class HBPNormedNdtrNLLLoss(HBPLoss):
    """Hessian backpropagation for the ``NormedNdtrNLLLoss`` layer."""

    def __init__(self):
        """Pass derivatives for ``NormedNdtrNLLLoss``."""
        super().__init__(derivatives=NormedNdtrNLLLossDerivatives())


class NormedNdtrNLLLoss(torch.nn.modules.loss._Loss):
    """Normed sigmoid NLL loss implementation."""

    def __init__(self):
        super().__init__()

        self.log_act_fn = log_normed_ndtr

    def forward(self, logit, target):
        return -self.log_act_fn(logit)[torch.arange(target.shape[0]), target].mean()


class NormedNdtrNLLLossDerivatives(NLLLossDerivatives):
    """Derivatives of the NormedNdtrNLLLoss."""

    def __init__(self):
        """Initialization for NormedNdtrNLLLoss derivative."""
        super().__init__(use_autograd=False)

    def _verify_support(self, module: NormedNdtrNLLLoss):
        """Verification of module support for NormedNdtrNLLLoss.

        Args:
            module: NormedNdtrNLLLoss module
        """
        self._check_input_dims(module)

    @staticmethod
    def _check_input_dims(module: NormedNdtrNLLLoss):
        """Raises an exception if the shapes of the input are not supported.

        Args:
            module: NormedNdtrNLLLoss module

        Raises:
            NotImplementedError: if input is not a batch of scalars.
        """
        if module.input0.dim() != 2:
            msg = "Only 2D inputs are currently supported"
            raise NotImplementedError(msg)

    @staticmethod
    def _make_distribution(subsampled_input: Tensor) -> Categorical:
        """Make the sampling distribution for the NormedNdtrNLLLoss.

        Args:
            subsampled_input: input after subsampling

        Returns:
            Categorical distribution with probabilities from the subsampled_input.
        """
        elementwise_log_probs = log_ndtr(subsampled_input)
        log_probs = elementwise_log_probs - torch.logsumexp(
            elementwise_log_probs, dim=1, keepdim=True
        )

        return Categorical(probs=log_probs.exp())

    @staticmethod
    def _get_mean_normalization(input: Tensor) -> int:
        return input.numel() // input.shape[1]

    @staticmethod
    def hessian_is_psd() -> bool:
        """Return whether the Hessian is PSD.

        Returns:
            True
        """
        return True

    def _compute_sampled_grads_manual(
        self, subsampled_input: torch.Tensor, mc_samples: int
    ) -> torch.Tensor:
        # probs
        subsampled_input = subsampled_input.double()
        log_probs = log_ndtr(subsampled_input)  # [N C D1 D2]

        # norm probs
        log_norm_factors = torch.logsumexp(
            log_probs, dim=1, keepdim=True
        )  # [N 1 D1 D2]

        # normal pdf
        normal = torch.distributions.Normal(0, 1)
        log_pdf_vals = normal.log_prob(subsampled_input)  # [N C D1 D2]

        first_term = (log_pdf_vals - log_norm_factors).exp()  # [N C D1 D2]

        # labels
        distribution = self._make_distribution(subsampled_input)
        samples = distribution.sample(torch.Size([mc_samples]))  # [V N D1 D2]
        samples_onehot = F.one_hot(
            samples, num_classes=log_probs.shape[1]
        )  # [V N D1 D2 C]
        samples_onehot_rearranged = torch.einsum("vn...c->vnc...", samples_onehot).to(
            log_probs.dtype
        )  # [V N C D1 D2]

        second_term = (
            log_pdf_vals - log_probs
        ).exp() * samples_onehot_rearranged  # [V N C D1 D2]

        return (first_term - second_term).float()
