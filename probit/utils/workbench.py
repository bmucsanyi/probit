"""Partial derivatives for cross-entropy loss."""

import torch
from backpack.extensions.secondorder.hbp.losses import (
    CrossEntropyLossDerivatives,
    HBPLoss,
)
from torch.nn.functional import log_softmax


class HBPMyCrossEntropyLoss(HBPLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives(use_autograd=True))


class MyCrossEntropyLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.ignore_index = -100
        self.weight = None

    def forward(self, x, y):
        indices = y != self.ignore_index
        x = x[indices]
        y = y[indices]
        x = log_softmax(x, dim=1)
        return -x[torch.arange(y.shape[0]), y].mean()


# class CrossEntropyLossDerivatives(NLLLossDerivatives):
#     """Partial derivatives for cross-entropy loss.

#     The `torch.nn.CrossEntropyLoss` operation is a composition of softmax
#     and negative log-likelihood.
#     """

#     def __init__(self, use_autograd: bool = False):
#         """Initialization for CE loss derivative.

#         Args:
#             use_autograd: Compute gradients with autograd (rather than manual)
#                 Defaults to ``False`` (manual computation).
#         """
#         super().__init__(use_autograd=use_autograd)

#     def _sqrt_hessian(
#         self,
#         module: CrossEntropyLoss,
#         g_inp: tuple[Tensor],
#         g_out: tuple[Tensor],
#         subsampling: list[int] = None,
#     ) -> Tensor:
#         raise NotImplementedError
#         # self._check_2nd_order_parameters(module)

#         # probs = self._get_probs(module, subsampling=subsampling)
#         # probs, *rearrange_info = self._merge_batch_and_additional(probs)

#         # tau = probs.sqrt()
#         # V_dim, C_dim = 0, 2
#         # Id = diag_embed(ones_like(probs), dim1=V_dim, dim2=C_dim)
#         # Id_tautau = Id - einsum("nv,nc->vnc", tau, tau)
#         # sqrt_H = einsum("nc,vnc->vnc", tau, Id_tautau)

#         # if module.reduction == "mean":
#         #     sqrt_H /= sqrt(self._get_mean_normalization(module.input0))

#         # sqrt_H = self._ungroup_batch_and_additional(sqrt_H, *rearrange_info)
#         # sqrt_H = self._expand_sqrt_h(sqrt_H)
#         # return sqrt_H

#     def _sum_hessian(
#         self, module: CrossEntropyLoss, g_inp: tuple[Tensor], g_out: tuple[Tensor]
#     ) -> Tensor:
#         raise NotImplementedError
#         # self._check_2nd_order_parameters(module)

#         # probs = self._get_probs(module)

#         # if probs.dim() == 2:
#         #     diagonal = diag(probs.sum(0))
#         #     sum_H = diagonal - einsum("nc,nd->cd", probs, probs)
#         # else:
#         #     out_shape = (*probs.shape[1:], *probs.shape[1:])
#         #     additional = probs.shape[2:].numel()

#         #     diagonal = diag(probs.sum(0).flatten()).reshape(out_shape)

#         #     probs = probs.flatten(2)
#         #     kron_delta = eye(additional, device=probs.device, dtype=probs.dtype)

#         #     sum_H = diagonal - einsum(
#         #         "ncx,ndy,xy->cxdy", probs, probs, kron_delta
#         #     ).reshape(out_shape)

#         # if module.reduction == "mean":
#         #     sum_H /= self._get_mean_normalization(module.input0)

#         # return sum_H

#     def _make_hessian_mat_prod(
#         self, module: CrossEntropyLoss, g_inp: tuple[Tensor], g_out: tuple[Tensor]
#     ) -> Callable[[Tensor], Tensor]:
#         self._check_2nd_order_parameters(module)

#         probs = self._get_probs(module)

#         def hessian_mat_prod(mat):
#             Hmat = einsum("...,v...->v...", probs, mat) - einsum(
#                 "nc...,nd...,vnd...->vnc...", probs, probs, mat
#             )

#             if module.reduction == "mean":
#                 Hmat /= self._get_mean_normalization(module.input0)

#             return Hmat

#         return hessian_mat_prod

#     def hessian_is_psd(self) -> bool:
#         """Return whether cross-entropy loss Hessian is positive semi-definite.

#         Returns:
#             True
#         """
#         return True

#     @staticmethod
#     def _get_probs(module: CrossEntropyLoss, subsampling: list[int] = None) -> Tensor:
#         """Compute the softmax probabilities from the module input.

#         Args:
#             module: cross-entropy loss with I/O.
#             subsampling: Indices of samples to be considered. Default of ``None`` uses
#                 the full mini-batch.

#         Returns:
#             Softmax probabilites
#         """
#         input0 = subsample(module.input0, subsampling=subsampling)
#         return softmax(input0, dim=1)


#     @staticmethod
#     def _merge_batch_and_additional(
#         probs: Tensor,
#     ) -> tuple[Tensor, str, dict[str, int]]:
#         """Rearranges the input if it has additional axes.

#         Treat additional axes like batch axis, i.e. group ``n c d1 d2 -> (n d1 d2) c``.

#         Args:
#             probs: the tensor to rearrange

#         Returns:
#             a tuple containing
#                 - probs: the rearranged tensor
#                 - str_d_dims: a string representation of the additional dimensions
#                 - d_info: a dictionary encoding the size of the additional dimensions
#         """
#         leading = 2
#         additional = probs.dim() - leading

#         str_d_dims: str = "".join(f"d{i} " for i in range(additional))
#         d_info: dict[str, int] = {
#             f"d{i}": probs.shape[leading + i] for i in range(additional)
#         }

#         probs = rearrange(probs, f"n c {str_d_dims} -> (n {str_d_dims}) c")

#         return probs, str_d_dims, d_info

#     @staticmethod
#     def _ungroup_batch_and_additional(
#         tensor: Tensor, str_d_dims, d_info, free_axis: int = 1
#     ) -> Tensor:
#         """Rearranges output if it has additional axes.

#         Used with group_batch_and_additional.

#         Undoes treating additional axes like batch axis and assumes an number of
#         additional free axes (``v``) were added, i.e. un-groups
#         ``v (n d1 d2) c -> v n c d1 d2``.

#         Args:
#             tensor: the tensor to rearrange
#             str_d_dims: a string representation of the additional dimensions
#             d_info: a dictionary encoding the size of the additional dimensions
#             free_axis: Number of free leading axes. Default: ``1``.

#         Returns:
#             the rearranged tensor

#         Raises:
#             NotImplementedError: If ``free_axis != 1``.
#         """
#         if free_axis != 1:
#             raise NotImplementedError(f"Only supports free_axis=1. Got {free_axis}.")

#         return rearrange(
#             tensor, f"v (n {str_d_dims}) c -> v n c {str_d_dims}", **d_info
#         )

#     @staticmethod
#     def _expand_sqrt_h(sqrt_h: Tensor) -> Tensor:
#         """Expands the square root hessian if CrossEntropyLoss has additional axes.

#         In the case of e.g. two additional axes (A and B), the input is [N,C,A,B].
#         In CrossEntropyLoss the additional axes are treated independently.
#         Therefore, the intermediate result has shape [C,N,C,A,B].
#         In subsequent calculations the additional axes are not independent anymore.
#         The required shape for sqrt_h_full is then [C*A*B,N,C,A,B].
#         Due to the independence, sqrt_h lives on the diagonal of sqrt_h_full.

#         Args:
#             sqrt_h: intermediate result, shape [C,N,C,A,B]

#         Returns:
#             sqrt_h_full, shape [C*A*B,N,C,A,B], sqrt_h on diagonal.
#         """
#         if sqrt_h.dim() > 3:
#             return diag_embed(sqrt_h.flatten(3), offset=0, dim1=1, dim2=4).reshape(
#                 -1, *sqrt_h.shape[1:]
#             )
#         return sqrt_h

#     @staticmethod
#     def _get_mean_normalization(input: Tensor) -> int:
#         """Get normalization constant used with reduction='mean'.

#         Args:
#             input: Input to the cross-entropy module.

#         Returns:
#             Divisor for mean reduction.
#         """
#         return input.numel() // input.shape[1]

#     def _verify_support(self, module: CrossEntropyLoss):
#         """We only support default weight and ignore_index.

#         Args:
#             module: CrossEntropyLoss module
#         """

#     def _make_distribution(self, subsampled_input: Tensor) -> Categorical:
#         """Create the likelihood distribution whose NLL is the CE.

#         The log probability of the Categorical distribution for a single sample
#         with k classes is ∑ᵢ₌₁ᵏ Ŷᵢ log pᵢ, where Ŷ is one-hot encoded. If p is
#         chosen as the softmax, this is equivalent to CrossEntropyLoss

#         Args:
#             subsampled_input: input after subsampling

#         Returns:
#             Normal distribution for targets | inputs
#         """
#         probs = softmax(subsampled_input, dim=1)
#         probs_rearranged = einsum("nc...->n...c", probs)
#         return Categorical(probs_rearranged)

#     def _compute_sampled_grads_manual(
#         self, subsampled_input: Tensor, mc_samples: int
#     ) -> Tensor:
#         """Manually compute gradients from sampled targets.

#         Cross Entropy loss is ∑ᵢ₌₁ᵏ Ŷᵢ log 𝜎(xᵢ), where 𝜎(xᵢ) is the softmax of
#         the input and Ŷᵢ is one-hot encoded. The gradient is 𝜎(xᵢ) - Ŷᵢ.

#         Args:
#             subsampled_input: input after subsampling
#             mc_samples: number of samples

#         Returns:
#             Gradient samples
#         """
#         raise NotImplementedError
