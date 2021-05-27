import torch
import numpy as np


def invF(x):
    """Inverse cumulative distribution function.
    
    Inverse cumulative distribution of a v-shape function,
    `f(x) = -4x+2 if x < .5 else 4x-2`.
    
    :param x: Number sequence between 0 and 1.
    :return: Inverse cumulative distribution evaluated at `x`.
    """
    y = np.ones_like(x) * 0.5

    sel = x < 0.5
    y[sel] *= 1.0 - np.sqrt(1.0 - 2.0 * x[sel])
    y[~sel] *= 1.0 + np.sqrt(2.0 * x[~sel] - 1.0)

    return y


def generate_data(n, *, std, seed):
    """Generates a synthetic data sample.
    
    Generates a 3D synthetic data sample with `n` data points.
    
    :param n: Number of data points.
    :param std: Standard deviation of Gaussian noise added to the data points.
    :param seed: Seed used for random number generation.
    :return: Synthetic data sample.
    """
    rng = np.random.default_rng(seed)

    t_flat = rng.uniform(size=n).astype(np.float32)
    t = invF(t_flat) * 2.0 * np.pi

    r = 1.0 + rng.normal(scale=std, size=n).astype(np.float32)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return torch.tensor([t, x, y]).T
