import collections

import torch
import torch.nn as nn

from model import Model as Base


class EDLActivation(nn.Module):
    """Tranforms the output of a 2D EDL model into NIW parameters.
    
    Tranforms the 6D output of a 2D EDL model into the parameters of a 2D NIW
    distribution.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input into NIW parameters.
        
        The input of batched 6D numbers is transformed into batched NIW parameters.
        
        :param x: Tensor of shape (batch_size, 6).
        :return: Tensor of shape (batch_size, 6).
        """
        return torch.stack(
            (
                x[:, 0],  # mu[0]
                x[:, 1],  # mu[1]
                torch.exp(x[:, 2]),  # L[0, 0]
                x[:, 3],  # L[1, 0]
                torch.exp(x[:, 4]),  # L[1, 1]
                10.0 * (torch.tanh(x[:, 5]) + 1.0) / 2.0 + 3.0,  # nu
            ),
            1,
        )


class EDL(nn.Module):
    """2D EDL model
    
    Transforms univariate input into the parameters of a 2D NIW distribution.
    """

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            collections.OrderedDict(
                [("model", Base(6)), ("edl_activation", EDLActivation()),]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
