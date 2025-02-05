"""Standard normal CDF utils."""

import math

import torch.nn.functional as F


def log_ndtr_approx(x):
    return F.logsigmoid(math.sqrt(8 / math.pi) * (x + 0.044715 * x**3))


def ndtr_approx(x):
    return F.sigmoid(math.sqrt(8 / math.pi) * (x + 0.044715 * x**3))
