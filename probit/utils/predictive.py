"""Predictive distribution utils."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from numpy import sqrt
from scipy.special import owens_t
from torch.distributions import Normal
from torch.special import log_ndtr, ndtr

from probit.utils.ndtr import log_ndtr_approx, ndtr_approx

LAMBDA_0 = math.pi / 8


def softmax_laplace_bridge(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    use_correction: bool = True,
    return_logits: bool = False,
) -> torch.Tensor:
    """Softmax + Laplace bridge predictive."""
    params = get_laplace_bridge_approximation(mean, var, use_correction=use_correction)
    pred = dirichlet_predictive(params)

    return pred.add(1e-10).log() if return_logits else pred


def softmax_mean_field(
    mean: torch.Tensor, var: torch.Tensor, *, return_logits: bool = False
) -> torch.Tensor:
    logits = mean / (1 + LAMBDA_0 * var).sqrt()

    return logits if return_logits else logits.softmax(dim=-1)


def softmax_mc(
    mean: torch.Tensor,
    var: torch.Tensor,
    num_mc_samples: int,
    *,
    return_samples: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    logit_samples = torch.randn(
        mean.shape[0],
        num_mc_samples,
        mean.shape[1],
        dtype=mean.dtype,
        device=mean.device,
    ) * var.sqrt().unsqueeze(1) + mean.unsqueeze(1)

    prob_mean = logit_samples.softmax(dim=-1).mean(dim=1)

    return (prob_mean, logit_samples) if return_samples else prob_mean


def logit_link_sigmoid_output(
    mean: torch.Tensor, var: torch.Tensor, *, return_logits: bool = False
) -> torch.Tensor:
    return probit_predictive(
        mean,
        var,
        link_function="logit",
        output_function="sigmoid",
        approximate=False,
        return_logits=return_logits,
    )


def logit_link_mc(
    mean: torch.Tensor,
    var: torch.Tensor,
    num_mc_samples: int,
    *,
    return_samples: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    logit_samples = torch.randn(
        mean.shape[0],
        num_mc_samples,
        mean.shape[1],
        dtype=mean.dtype,
        device=mean.device,
    ) * var.sqrt().unsqueeze(1) + mean.unsqueeze(1)
    prob = F.sigmoid(logit_samples)
    prob = prob / prob.sum(dim=-1, keepdim=True)

    prob_mean = prob.mean(dim=1)

    return (prob_mean, logit_samples) if return_samples else prob_mean


def probit_link_normcdf_output(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    approximate: bool,
    return_logits: bool = False,
) -> torch.Tensor:
    return probit_predictive(
        mean,
        var,
        link_function="probit",
        output_function="normcdf",
        approximate=approximate,
        return_logits=return_logits,
    )


def log_link(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    return_logits: bool = False,
) -> torch.Tensor:
    return probit_predictive(
        mean,
        var,
        link_function="log",
        output_function=None,
        approximate=False,
        return_logits=return_logits,
    )


def log_link_mc(
    mean: torch.Tensor,
    var: torch.Tensor,
    num_mc_samples: int,
    *,
    return_samples: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    logit_samples = torch.randn(
        mean.shape[0],
        num_mc_samples,
        mean.shape[1],
        dtype=mean.dtype,
        device=mean.device,
    ) * var.sqrt().unsqueeze(1) + mean.unsqueeze(1)  # [B, S, C]
    prob = F.softmax(logit_samples, dim=-1)

    prob_mean = prob.mean(dim=1)

    return (prob_mean, logit_samples) if return_samples else prob_mean


def logit_link_sigmoid_output_dirichlet(
    mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    return get_mom_dirichlet_approximation(
        mean,
        var,
        link_function="logit",
        output_function="sigmoid",
        approximate=False,
    )


def logit_link_sigmoid_product_output_dirichlet(
    mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    return get_mom_dirichlet_approximation(
        mean,
        var,
        link_function="logit",
        output_function="sigmoid_product",
        approximate=False,
    )


def probit_link_normcdf_output_dirichlet(
    mean: torch.Tensor,
    var: torch.Tensor,
    approximate: bool,
) -> torch.Tensor:
    return get_mom_dirichlet_approximation(
        mean,
        var,
        link_function="probit",
        output_function="normcdf",
        approximate=approximate,
    )


def log_link_dirichlet(
    mean: torch.Tensor,
    var: torch.Tensor,
) -> torch.Tensor:
    return get_mom_dirichlet_approximation(
        mean,
        var,
        link_function="log",
        output_function="",
        approximate=False,
    )


def softmax_laplace_bridge_dirichlet(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    use_correction: bool = True,
) -> torch.Tensor:
    params = get_laplace_bridge_approximation(mean, var, use_correction=use_correction)

    return params


def probit_link_mc(
    mean: torch.Tensor,
    var: torch.Tensor,
    num_mc_samples: int,
    *,
    approximate: bool,
    return_samples: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    logit_samples = torch.randn(
        mean.shape[0],
        num_mc_samples,
        mean.shape[1],
        dtype=mean.dtype,
        device=mean.device,
    ) * var.sqrt().unsqueeze(1) + mean.unsqueeze(1)

    ndtr_fn = ndtr_approx if approximate else ndtr
    prob = ndtr_fn(logit_samples.double()).float()
    prob = prob / prob.sum(dim=-1, keepdim=True)

    prob_mean = prob.mean(dim=1)

    return (prob_mean, logit_samples) if return_samples else prob_mean


def probit_predictive(
    mean: torch.Tensor,
    var: torch.Tensor,
    link_function: str = "probit",
    output_function: str = "normcdf",
    *,
    approximate: bool,
    return_logits: bool = False,
) -> torch.Tensor:
    """Predictive distribution with the probit link function or approximation."""
    predictives = gaussian_pushforward_mean(
        mean,
        var,
        link_function,
        output_function,
        approximate=approximate,
        return_logits=return_logits,
    )  # [batch_size, num_classes]

    if return_logits:
        return predictives

    sum_predictives = torch.sum(predictives, dim=1, keepdim=True)  # [batch_size, 1]
    predictives = predictives / sum_predictives  # [batch_size, num_classes]

    return predictives


def beta_predictive(beta_params: torch.Tensor) -> torch.Tensor:
    """Predictive mean of beta distributions."""
    predictives = beta_params[:, :, 0] / torch.sum(
        beta_params, dim=2, keepdim=False
    )  # [batch_size, num_classes]
    sum_predictives = torch.sum(
        predictives, dim=1, keepdim=True
    )  # [batch_size, num_classes]
    norm_predictives = predictives / sum_predictives  # [batch_size, num_classes]
    return norm_predictives


def dirichlet_predictive(params: torch.Tensor) -> torch.Tensor:
    """Predictive mean of Dirichlet distributions."""
    predictives = params / torch.sum(
        params, dim=1, keepdim=True
    )  # [batch_size, num_classes]

    # remove nans due to infinite dirichlet parameters
    is_nan = torch.isnan(predictives)
    nan_count = torch.sum(is_nan, dim=1, keepdim=True)
    predictives = torch.where(
        torch.any(is_nan, dim=1, keepdim=True),
        torch.zeros(predictives.shape, device=predictives.device),
        predictives,
    )
    predictives = torch.where(
        is_nan, torch.ones_like(predictives) / nan_count, predictives
    )

    return predictives


def get_laplace_bridge_approximation(
    mean: torch.Tensor, var: torch.Tensor, *, use_correction: bool = True
) -> torch.Tensor:
    """Laplace bridge approximation."""
    num_classes = mean.shape[1]

    if use_correction:
        c = torch.sum(var, dim=1) * (1 / sqrt(num_classes / 2))  # [B]
        c_expanded = torch.tile(c[:, None], (1, num_classes))  # [B, C]
        mean_p = mean / torch.sqrt(c_expanded)  # [B, C]
        var_p = var / c_expanded  # [B, C]
    else:
        mean_p = mean
        var_p = var

    # Laplace bridge
    sum_exp_neg_mean_p = torch.sum(torch.exp(-mean_p), dim=1)  # [batch_size]
    sum_exp_neg_mean_p_expanded = torch.tile(
        sum_exp_neg_mean_p[:, None], (1, num_classes)
    )  # [B, C]
    params = (
        1
        - 2 / num_classes
        + torch.exp(mean_p) * sum_exp_neg_mean_p_expanded / (num_classes**2)
    ) / var_p

    return params


def probit_scale(link_function: str = "probit") -> float:
    if link_function == "probit":
        scale = 1.0
    elif link_function == "logit":
        scale = LAMBDA_0
    else:
        error_message = "Invalid link function"
        raise NotImplementedError(error_message)
    return scale


def gaussian_pushforward_mean(
    means: torch.Tensor,
    vars: torch.Tensor,
    link_function: str = "probit",
    output_function: str = "normcdf",
    *,
    approximate: bool,
    return_logits: bool = False,
) -> torch.Tensor:
    if link_function == "log":
        if return_logits:
            return means + vars / 2
        return torch.exp(means + vars / 2)

    if link_function in {"probit", "logit"}:
        scale = probit_scale(link_function)
        if "normcdf" in output_function:
            logits = means / torch.sqrt(1 / scale + vars)

            if return_logits:
                return logits

            ndtr_fn = ndtr_approx if approximate else ndtr
            predictives = ndtr_fn(logits.double()).float()
        elif "sigmoid" in output_function:
            logits = means / torch.sqrt(1 + scale * vars)

            if return_logits:
                return logits

            predictives = F.sigmoid(logits)
        else:
            msg = "Invalid output function"
            raise NotImplementedError(msg)
    else:
        msg = "Invalid link function"
        raise NotImplementedError(msg)

    return predictives


def gaussian_pushforward_second_moment(
    means: torch.Tensor,
    vars: torch.Tensor,
    link_function: str = "probit",
    output_function: str = "normcdf",
    *,
    approximate: bool,
) -> torch.Tensor:
    if link_function == "log":
        return torch.exp(2 * means + 2 * vars)

    scale = probit_scale(link_function)

    if link_function in {"probit", "logit"}:
        if output_function == "sigmoid_product":
            if link_function == "logit":
                s = gaussian_pushforward_mean(
                    means, vars, "logit", "sigmoid", approximate=False
                )

                second_moment = s - s * (1 - s) / torch.sqrt(
                    1 + scale * vars
                )  # = s * (1 - s) * (1 - 1 / torch.sqrt(1 + scale * vars)) + s**2
            else:
                msg = "Invalid link function for sigmoid_product"
                raise NotImplementedError(msg)
        else:
            device = means.device

            owens_t_input1 = (means / torch.sqrt(1 / scale + vars)).cpu().numpy()
            owens_t_input2 = (1 / torch.sqrt(1 + 2 * scale * vars)).cpu().numpy()

            t_term = -2 * torch.from_numpy(owens_t(owens_t_input1, owens_t_input2)).to(
                device
            )
            p_term = gaussian_pushforward_mean(
                means, vars, link_function, output_function, approximate=approximate
            )
            second_moment = p_term + t_term
    else:
        msg = "Invalid link function"
        raise NotImplementedError

    return second_moment


def get_mom_beta_approximation(
    means: torch.Tensor,
    vars: torch.Tensor,
    link_function: str = "probit",
    output_function: str = "normcdf",
    *,
    approximate: bool,
) -> torch.Tensor:
    M1 = gaussian_pushforward_mean(
        means, vars, link_function, output_function, approximate=approximate
    )
    M2 = gaussian_pushforward_second_moment(
        means, vars, link_function, output_function, approximate=approximate
    )

    beta_params = torch.ones((*means.shape, 2), device=means.device)
    L = (M1 - M2) / (M2 - M1**2)
    beta_params[..., 0] = M1 * L
    beta_params[..., 1] = (1 - M1) * L
    return beta_params


def get_mom_dirichlet_approximation(
    means: torch.Tensor,
    vars: torch.Tensor,
    link_function: str = "probit",
    output_function: str = "normcdf",
    *,
    approximate: bool,
) -> torch.Tensor:
    M1 = gaussian_pushforward_mean(
        means, vars, link_function, output_function, approximate=approximate
    )
    M2 = gaussian_pushforward_second_moment(
        means, vars, link_function, output_function, approximate=approximate
    )
    S1 = torch.sum(M1, dim=-1, keepdim=True)
    S = torch.maximum(S1, torch.ones(S1.shape, device=S1.device))
    P_EPS = 1e-10
    LP = torch.mean(
        torch.log(
            torch.clamp((M1 * S - M2) / torch.clamp(M2 - M1**2, min=P_EPS), min=P_EPS)
        ),
        dim=-1,
        keepdim=True,
    )
    P = torch.exp(LP)
    return P * M1 / S


@torch.compile
def diag_hessian_normalized_sigmoid(logit, target):
    q = torch.sigmoid(logit)
    p = q / q.sum(dim=-1, keepdim=True)

    y = F.one_hot(target, num_classes=logit.shape[1]).to(logit.dtype)

    q_complement = 1 - q
    pq_complement = p * q_complement

    return y * q * q_complement + p * (1 - 3 * q + 2 * q**2) - pq_complement**2


def diag_hessian_softmax(logit, target):
    del target
    prob = logit.softmax(dim=-1)  # [B, C]

    return prob * (1 - prob)  # [B, C]


@torch.compile
def diag_hessian_normalized_normcdf(logit, target, approximate):
    ndtr_fn = ndtr_approx if approximate else ndtr
    q = ndtr_fn(logit.double()).float()
    s = q.sum(dim=-1, keepdim=True)
    normal = Normal(0, 1)
    phi = normal.log_prob(logit).exp()  # Norm pdf
    theta = -logit * phi  # Norm pdf derivative

    y = F.one_hot(target, num_classes=logit.shape[1]).to(logit.dtype)

    return theta * (1 / s - y / q) + phi**2 * (y / q**2 - 1 / s**2)


PREDICTIVE_DICT = {
    "softmax_laplace_bridge": softmax_laplace_bridge,
    "softmax_mean_field": softmax_mean_field,
    "softmax_mc": softmax_mc,
    "logit_link_sigmoid_output": logit_link_sigmoid_output,
    "logit_link_sigmoid_product_output": logit_link_sigmoid_output,
    "logit_link_mc": logit_link_mc,
    "probit_link_normcdf_output": probit_link_normcdf_output,
    "probit_link_mc": probit_link_mc,
    "log_link": log_link,
    "log_link_mc": log_link_mc,
}


def get_predictive(predictive, use_correction, num_mc_samples, approximate):
    predictive_fn = PREDICTIVE_DICT[predictive]

    if predictive.endswith("mc"):
        predictive_fn = partial(predictive_fn, num_mc_samples=num_mc_samples)
    elif predictive == "softmax_laplace_bridge":
        predictive_fn = partial(predictive_fn, use_correction=use_correction)

    if predictive.startswith("probit"):
        predictive_fn = partial(predictive_fn, approximate=approximate)

    return predictive_fn


DIRICHLET_DICT = {
    "logit_link_sigmoid_output": logit_link_sigmoid_output_dirichlet,
    "logit_link_sigmoid_product_output": logit_link_sigmoid_product_output_dirichlet,
    "probit_link_normcdf_output": probit_link_normcdf_output_dirichlet,
    "log_link": log_link_dirichlet,
    "softmax_laplace_bridge": softmax_laplace_bridge_dirichlet,
}


def get_dirichlet(dirichlet_str, approximate, use_correction):
    dirichlet_fn = DIRICHLET_DICT[dirichlet_str]

    if dirichlet_str.startswith("probit"):
        dirichlet_fn = partial(dirichlet_fn, approximate=approximate)
    elif dirichlet_str.startswith("softmax"):
        dirichlet_fn = partial(dirichlet_fn, use_correction=use_correction)

    return dirichlet_fn


def normed_sigmoid(x, *, unnormalized=False):
    x = F.sigmoid(x)

    if unnormalized:
        return x

    x = x / x.sum(dim=-1, keepdim=True)

    return x


def log_normed_sigmoid(x):
    x = F.logsigmoid(x)

    return x - torch.logsumexp(x, dim=-1, keepdim=True)


def normed_ndtr_approx(x, *, unnormalized=False):
    x = ndtr_approx(x)

    if unnormalized:
        return x

    x = x / x.sum(dim=-1, keepdim=True)

    return x


def normed_exp(x, *, unnormalized=False):
    if unnormalized:
        return torch.exp(x)

    return torch.softmax(x, dim=-1)


def log_normed_ndtr_approx(x):
    x = log_ndtr_approx(x)  # [B, C]

    return x - torch.logsumexp(x, dim=-1, keepdim=True)


def normed_ndtr(x, *, unnormalized=False):
    x = ndtr(x)

    if unnormalized:
        return x

    x = x / x.sum(dim=-1, keepdim=True)

    return x


def log_normed_ndtr(x):
    x = log_ndtr(x)

    return x - torch.logsumexp(x, dim=-1, keepdim=True)


def log_normed_exp(x):
    return F.log_softmax(x, dim=-1)


def get_activation(predictive, approximate, *, unnormalized=False):
    if predictive.startswith("softmax") and unnormalized:
        msg = "Invalid parameters provided"
        raise ValueError(msg)

    if predictive.startswith("softmax"):
        return partial(F.softmax, dim=-1)

    if predictive.startswith("log_"):
        return partial(normed_exp, unnormalized=unnormalized)

    if predictive.startswith("probit"):
        fn = normed_ndtr_approx if approximate else normed_ndtr
    elif predictive.startswith("logit"):
        fn = normed_sigmoid
    else:
        msg = "Invalid predictive provided"
        raise ValueError(msg)

    fn = partial(fn, unnormalized=unnormalized)

    return fn


def get_log_activation(predictive, approximate):
    if predictive.startswith(("softmax", "log_")):
        return log_normed_exp
    if predictive.startswith("probit"):
        return log_normed_ndtr_approx if approximate else log_normed_ndtr
    if predictive.startswith("logit"):
        return log_normed_sigmoid

    msg = "Invalid predictive provided"
    raise ValueError(msg)
