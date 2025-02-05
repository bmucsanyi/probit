"""KFAC covariance pushforward Laplace wrapper."""

import gc
import logging
import time

import numpy as np
import torch
from backpack import backpack, extend
from backpack.extensions import KFAC

from probit.losses.normed_ndtr_loss import HBPNormedNdtrNLLLoss, NormedNdtrNLLLoss
from probit.losses.normed_sigmoid_loss import (
    HBPNormedSigmoidNLLLoss,
    NormedSigmoidNLLLoss,
)
from probit.utils.metric import calibration_error
from probit.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class CovariancePushforwardLLLaplaceWrapper(DistributionalWrapper):
    """KFAC covariance pushforward last-layer Laplace wrapper."""

    def __init__(
        self,
        model,
        loss_fn,
        predictive_fn,
        last_layer_name,
        prior_precision,
        ggn_scaler,
        weight_path,
    ):
        super().__init__(model)
        self._load_model(weight_path)

        setattr(
            self.model, last_layer_name, extend(getattr(self.model, last_layer_name))
        )

        self.last_layer_name = last_layer_name
        self.loss_fn = extend(loss_fn)
        self.extension = KFAC()

        if isinstance(loss_fn, NormedSigmoidNLLLoss):
            self.extension.set_module_extension(
                NormedSigmoidNLLLoss, HBPNormedSigmoidNLLLoss()
            )
        elif isinstance(loss_fn, NormedNdtrNLLLoss):
            self.extension.set_module_extension(
                NormedNdtrNLLLoss, HBPNormedNdtrNLLLoss()
            )

        self.prior_precision = prior_precision
        self.ggn_scaler = ggn_scaler

        self.predictive_fn = predictive_fn
        self.is_laplace_approximated = False
        self.apply_parameter_mask()

        print(
            "Number of Laplace-approximated parameters:",
            sum(
                param.numel()
                for param in self.model.parameters()
                if param.requires_grad
            ),
        )

    def forward(self, x):
        if not self.is_laplace_approximated:
            msg = "Model has to be Laplace approximated first"
            raise ValueError(msg)

        pre_logits = self.model.forward_head(
            self.model.forward_features(x), pre_logits=True
        )  # [B, D]
        diag_A, B = self.covariance_kfac[0]  # [C], [D, D]
        diag_C = self.covariance_kfac[1][0]  # [C]
        multipliers = ((pre_logits @ B) * pre_logits).sum(dim=1)  # [B]

        mean = self.model.get_classifier()(pre_logits)  # [B, C]
        var = multipliers.unsqueeze(1) * diag_A + diag_C  # [B, C]

        return mean, var

    def apply_parameter_mask(self):
        for param_name, param in self.model.named_parameters():
            if not param_name.startswith(self.last_layer_name):
                param.requires_grad = False

    def perform_laplace_approximation(
        self,
        train_loader,
        val_loader,
        channels_last,
        log_prior_prec_min=-1,
        log_prior_prec_max=2,
        grid_size=50,
    ):
        self.theta_0_list = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        self.theta_0_vec = torch.nn.utils.parameters_to_vector(
            self.theta_0_list
        ).detach()
        self.kfac = self.get_kfac_loader(train_loader, channels_last)
        self.is_laplace_approximated = True

        if self.prior_precision is None:
            self.prior_precision = self.optimize_prior_precision_cv(
                val_loader,
                channels_last,
                log_prior_prec_min,
                log_prior_prec_max,
                grid_size,
            )

        self.covariance_kfac = self.get_covariance_kfac(self.kfac, self.prior_precision)

    @staticmethod
    def get_ece(out_dist, targets):
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def optimize_prior_precision_cv(
        self,
        val_loader,
        channels_last,
        log_prior_prec_min=-1,
        log_prior_prec_max=2,
        grid_size=50,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        prior_precision = self.gridsearch(
            interval=interval,
            val_loader=val_loader,
            channels_last=channels_last,
        )

        logger.info(f"Optimized prior precision is {prior_precision}.")

        return prior_precision

    def gridsearch(
        self,
        interval,
        val_loader,
        channels_last,
    ):
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()

            try:
                self.prior_precision = prior_prec
                self.covariance_kfac = self.get_covariance_kfac(
                    self.kfac, self.prior_precision
                )

                out_dist, targets = self.validate(
                    val_loader=val_loader,
                    channels_last=channels_last,
                )
                result = self.get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()

                result = np.nan_to_num(result, nan=float("inf"))
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy {accuracy}"
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def validate(self, val_loader, channels_last):
        self.model.eval()
        device = next(self.model.parameters()).device
        output_means = []
        targets = []

        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            if channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            mean, var = self(input)
            out = self.predictive_fn(mean, var)

            if out.device.type == "cuda":
                torch.cuda.synchronize()

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def get_kfac_loader(self, train_loader, channels_last):
        """Compute the KFAC approximation based on a list of mini-batches `datalist`."""
        # Accumulate KFAC approximations over all mini-batches
        for i, (X, y) in enumerate(train_loader):
            num_total = 0
            if i % 500 == 0:
                print(f"Processing batch {i + 1}...")
            batch_size = X.shape[0]

            if i == 0:
                # Initialize `kfac`.
                # If there is only one mini-batch, return the KFAC approximation.
                kfac = self.get_kfac_minibatch(X, y, channels_last)

                if len(train_loader) == 1:
                    return kfac
            else:
                # Compute mini-batch KFAC approximation
                mb_kfac = self.get_kfac_minibatch(X, y, channels_last)

                # Add mini-batch KFAC approximation
                kfac = self.add_kfacs(
                    kfac,
                    mb_kfac,
                    num_elem1=num_total,
                    num_elem2=batch_size,
                )

                # Free VRAM
                del mb_kfac
                torch.cuda.empty_cache()
                gc.collect()

            num_total += batch_size

        return kfac

    @staticmethod
    def get_covariance_kfac(kfac_list, prior_precision):
        covariance_kfac_list = []
        for kfac in kfac_list:
            if len(kfac) == 2:
                A, B = kfac
                s_A, U_A = CovariancePushforwardLLLaplaceWrapper.get_eigendecomposition(
                    A
                )
                s_B, U_B = CovariancePushforwardLLLaplaceWrapper.get_eigendecomposition(
                    B
                )

                A_cov = (
                    U_A @ torch.diag(torch.reciprocal(s_A + prior_precision)) @ U_A.T
                )
                B_cov = (
                    U_B @ torch.diag(torch.reciprocal(s_B + prior_precision)) @ U_B.T
                )
                covariance_kfac_list.append((torch.diag(A_cov), B_cov))
            else:
                C = kfac[0]
                s_C, U_C = CovariancePushforwardLLLaplaceWrapper.get_eigendecomposition(
                    C
                )
                C_cov = (
                    U_C @ torch.diag(torch.reciprocal(s_C + prior_precision)) @ U_C.T
                )
                covariance_kfac_list.append((torch.diag(C_cov),))

        return covariance_kfac_list

    @staticmethod
    def get_eigendecomposition(M):
        """Get the eigendecomposition of a symmetric, PSD matrix `M`."""
        # Compute eigendecomposition
        eigvals, U = torch.linalg.eigh(M)

        return eigvals, U

    def get_kfac_minibatch(self, X, y, channels_last):
        """Get the KFAC approximation based on one mini-batch `(X, y)`.

        This returns a list-representation of KFAC. Its entries are lists that contain
        either a single matrix `[Fi]` or a pair of matrices `[Ai, Bi]` such that
        `Fi = Ai ⊗ Bi`. An example could look like this:
        ```
        kfac = [
            [F1],
            [A2, B2],  # F2 = A2 ⊗ B2
            [F3],
            [A4, B4],  # F4 = A4 ⊗ B4
            [A5, B5],  # F5 = A5 ⊗ B5
            ...
        ]
        ```
        """
        # Extend model and loss function. The use_converter parameter is used for ResNet
        # compatibility
        device = next(self.model.parameters()).device

        # Forward and backward pass
        X, y = X.to(device), y.to(device)
        batch_size = X.shape[0]

        if channels_last:
            X = X.contiguous(memory_format=torch.channels_last)

        with torch.enable_grad():
            loss = self.loss_fn(self.model(X), y)

            with backpack(self.extension):
                loss.backward()

            # Extract KFAC matrix from model
            kfac = []
            for param in self.model.parameters():
                if param.requires_grad:
                    param_factors = []
                    first_factor = param.kfac[0].detach()
                    param_factors.append(batch_size * first_factor)

                    if len(param.kfac) == 2:
                        second_factor = param.kfac[1].detach()
                        param_factors.append(second_factor)

                    kfac.append(param_factors)

        # Free GPU memory
        del loss
        torch.cuda.empty_cache()
        gc.collect()

        return CovariancePushforwardLLLaplaceWrapper.scale_kfac(kfac, self.ggn_scaler)

    @staticmethod
    def scale_kfac(kfac, scaler):
        """Scales a KFAC approximation."""
        scaled_kfac = []
        for param in kfac:
            scaled_param = []
            scaled_kfac.append(scaled_param)
            B = param[0]
            scaled_param.append(scaler * B)

            if len(param) == 2:
                A = param[1]
                scaled_param.append(A)

        return scaled_kfac

    @staticmethod
    def add_kfacs(kfac1, kfac2, num_elem1, num_elem2):
        """Add two KFAC approximations."""
        kfac3 = []
        for param1, param2 in zip(kfac1, kfac2, strict=True):
            param3 = []
            kfac3.append(param3)
            B1 = param1[0]
            B2 = param2[0]
            param3.append(B1 + B2)

            if len(param1) == 2:
                A1 = param1[1]
                A2 = param2[1]
                num_elem_total = num_elem1 + num_elem2
                param3.append(
                    num_elem1 / num_elem_total * A1 + num_elem2 / num_elem_total * A2
                )

        return kfac3
