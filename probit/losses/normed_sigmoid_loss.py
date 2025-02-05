"""Backpack extension for NormedSigmoidNLLLoss."""

import torch
import torch.nn.functional as F
from backpack.core.derivatives.nll_base import NLLLossDerivatives
from backpack.extensions.secondorder.hbp.losses import HBPLoss
from torch import Tensor
from torch.distributions import Categorical

from probit.utils.predictive import log_normed_sigmoid


class HBPNormedSigmoidNLLLoss(HBPLoss):
    """Hessian backpropagation for the ``NormedSigmoidNLLLoss`` layer."""

    def __init__(self):
        """Pass derivatives for ``NormedSigmoidNLLLoss``."""
        super().__init__(derivatives=NormedSigmoidNLLLossDerivatives())


class NormedSigmoidNLLLoss(torch.nn.modules.loss._Loss):
    """Normed sigmoid NLL loss implementation."""

    def __init__(self):
        super().__init__()

        self.log_act_fn = log_normed_sigmoid

    def forward(self, logit, target):
        return -self.log_act_fn(logit)[torch.arange(target.shape[0]), target].mean()


class NormedSigmoidNLLLossDerivatives(NLLLossDerivatives):
    """Derivatives of the NormedSigmoidNLLLoss."""

    def __init__(self):
        """Initialization for NormedSigmoidNLLLoss derivative."""
        super().__init__(use_autograd=False)

    def _verify_support(self, module: NormedSigmoidNLLLoss):
        """Verification of module support for NormedSigmoidNLLLoss.

        Args:
            module: NormedSigmoidNLLLoss module
        """
        self._check_input_dims(module)

    @staticmethod
    def _check_input_dims(module: NormedSigmoidNLLLoss):
        """Raises an exception if the shapes of the input are not supported.

        Args:
            module: NormedSigmoidNLLLoss module

        Raises:
            NotImplementedError: if input is not a batch of scalars.
        """
        if module.input0.dim() != 2:
            msg = "Only 2D inputs are currently supported"
            raise NotImplementedError(msg)

    @staticmethod
    def _make_distribution(subsampled_input: Tensor) -> Categorical:
        """Makes the sampling distribution for NormedSigmoidNLLLoss.

        Args:
            subsampled_input: input after subsampling

        Returns:
            Categorical distribution with probabilities from the subsampled_input.
        """
        elementwise_probs = subsampled_input.sigmoid()
        probs = elementwise_probs / elementwise_probs.sum(dim=1, keepdim=True)

        return Categorical(probs=probs)

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
        probs = torch.sigmoid(subsampled_input)
        expand_dims = [mc_samples] + probs.dim() * [-1]
        probs_unsqeezed = probs.unsqueeze(0).expand(*expand_dims)  # [V N C D1 D2]

        # norm probs
        norm_probs = probs / probs.sum(dim=1, keepdim=True)
        norm_probs_unsqeezed = norm_probs.unsqueeze(0).expand(
            *expand_dims
        )  # [V N C D1 D2]

        # labels
        distribution = self._make_distribution(subsampled_input)
        samples = distribution.sample(torch.Size([mc_samples]))  # [V N D1 D2]
        samples_onehot = F.one_hot(samples, num_classes=probs.shape[1])  # [V N D1 D2 C]
        samples_onehot_rearranged = torch.einsum("vn...c->vnc...", samples_onehot).to(
            probs.dtype
        )  # [V N C D1 D2]

        return (1 - probs_unsqeezed) * (
            norm_probs_unsqeezed - samples_onehot_rearranged
        )
