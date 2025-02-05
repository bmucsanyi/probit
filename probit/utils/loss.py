"""Loss function utilities."""

from torch import nn

from probit.losses import (
    BMACrossEntropyLoss,
    EDLLoss,
    ExpNLLLoss,
    NormCDFNLLLoss,
    RegularizedBMACrossEntropyLoss,
    RegularizedPredictiveNLLLoss,
    RegularizedSigmoidNLLLoss,
    RegularizedSoftmaxNLLLoss,
    RegularizedUCELoss,
    SigmoidNLLLoss,
    SoftmaxPredictiveNLLLoss,
    UnnormalizedPredictiveNLLLoss,
)
from probit.losses.normed_ndtr_loss import NormedNdtrNLLLoss
from probit.losses.normed_sigmoid_loss import NormedSigmoidNLLLoss


def get_laplace_loss_fn(predictive):
    if predictive.startswith(("softmax", "log_link")):
        return nn.CrossEntropyLoss()
    if predictive.startswith("probit"):
        return NormedNdtrNLLLoss()
    if predictive.startswith("logit"):
        return NormedSigmoidNLLLoss()

    msg = "Invalid predictive provided"
    raise ValueError(msg)


def create_loss_fn(args, num_batches):
    # Setup loss function
    if args.loss == "cross-entropy":
        train_loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "bma-cross-entropy":
        train_loss_fn = BMACrossEntropyLoss(
            predictive=args.predictive,
            use_correction=args.use_correction,
            num_mc_samples=args.num_mc_samples,
        )
    elif args.loss == "regularized-bma-cross-entropy":
        train_loss_fn = RegularizedBMACrossEntropyLoss(
            regularization_factor=args.regularization_factor,
            target_normalization_value=args.target_normalization_value,
        )
    elif args.loss == "edl":
        train_loss_fn = EDLLoss(
            num_batches=num_batches,
            num_classes=args.num_classes,
            start_epoch=args.edl_start_epoch,
            scaler=args.edl_scaler,
        )
    elif args.loss == "uce":
        train_loss_fn = RegularizedUCELoss(
            regularization_factor=args.regularization_factor
        )
    elif args.loss == "normcdf-nll":
        train_loss_fn = NormCDFNLLLoss(args.approximate)
    elif args.loss == "sigmoid-nll":
        train_loss_fn = SigmoidNLLLoss()
    elif args.loss == "exp-nll":
        train_loss_fn = ExpNLLLoss()
    elif args.loss == "regularized-softmax-nll":
        train_loss_fn = RegularizedSoftmaxNLLLoss(
            regularization_factor=args.regularization_factor,
            target_normalization_value=args.target_normalization_value,
        )
    elif args.loss == "regularized-sigmoid-nll":
        train_loss_fn = RegularizedSigmoidNLLLoss(
            regularization_factor=args.regularization_factor,
            target_normalization_value=args.target_normalization_value,
        )
    elif args.loss == "regularized-predictive-nll":
        train_loss_fn = RegularizedPredictiveNLLLoss(
            predictive=args.predictive,
            use_correction=args.use_correction,
            num_mc_samples=args.num_mc_samples,
            regularization_factor=args.regularization_factor,
            approximate=args.approximate,
            target_normalization_value=args.target_normalization_value,
        )
    elif args.loss == "unnormalized-predictive-nll":
        train_loss_fn = UnnormalizedPredictiveNLLLoss(
            predictive=args.predictive, approximate=args.approximate
        )
    elif args.loss == "softmax-predictive-nll":
        train_loss_fn = SoftmaxPredictiveNLLLoss(predictive=args.predictive)
    else:
        msg = f"--loss {args.loss} is not implemented"
        raise NotImplementedError(msg)

    return train_loss_fn
