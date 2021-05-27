import collections

import torch
import torch.nn as nn


class Model(nn.Module):
    """Simple fully conntected NN with 2 hidden layers.
    """

    def __init__(self, n_out=1):
        super().__init__()
        self.f = nn.Sequential(
            collections.OrderedDict(
                [
                    ("fc1", nn.Linear(1, 32)),
                    ("relu1", nn.ReLU()),
                    ("fc2", nn.Linear(32, 32)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(32, n_out)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x.unsqueeze(1)).squeeze(1)
