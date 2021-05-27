from enum import Enum

import torch
from torch import diagonal, lgamma, log, logdet, matmul
import numpy as np


def student_nll(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a multivariate t-distribution.
    
    :param x: Batched paramaters of a NIW distribution as tensor with shape
              (batch_size, n * (n + 3) / 2 + 1).
    :param y: Batched measurements as tensor with shape (batch_size, n).
    :return: Negative log-likelihood of x and y as tensor of shape (batch_size).
    """
    n = int(np.rint(-3.0 / 2.0 + np.sqrt(9.0 / 4.0 + 2.0 * (x.shape[1] - 1.0))))
    nu_idx = (n * (n + 3)) // 2

    mu = x[:, :n]

    idx = torch.tril_indices(n, n)
    L = torch.zeros(x.shape[0], n, n, dtype=x.dtype)
    L[:, idx[0], idx[1]] = x[:, n:nu_idx]
    sigma = matmul(L, L.transpose(1, 2))

    nu = x[:, nu_idx]

    k = 1.0 + nu
    d = y - mu

    ddT_over_k = matmul(d.unsqueeze(2), d.unsqueeze(1)) / k.unsqueeze(1).unsqueeze(2)

    if n == 2:
        nrm = -log(nu - 1)
    else:
        nrm = lgamma((nu - n + 1.0) / 2.0) - lgamma((nu + 1.0) / 2.0)

    return (
        nrm
        + n / 2.0 * log(k)
        - nu * torch.sum(log(diagonal(L, dim1=-2, dim2=-1)), dim=1)
        + (nu + 1) / 2.0 * logdet(sigma + ddT_over_k)
    )


def student_nll_mean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean of negative log-likelihoods of multivariate t-distributions.
    
    :param x: Input as passed as `x` to `student_nll`.
    :param y: Input as passed as `y` to `student_nll`.
    :return: Mean of the result of `student_nll(x, y)`.
    """
    return torch.mean(student_nll(x, y))
