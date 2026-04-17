import torch
import torch.nn as nn


class WindowedMLP(nn.Module):
    """Windowed MLP (Hwangbo et al., 2019).

    Accepts a 3-D tensor (batch, seq_len, n_features), flattens the last two
    dimensions, and passes the result through `n_layers` hidden layers of
    `hidden_size` units with Softsign activations, followed by a linear
    output layer that predicts a scalar torque value.
    """

    def __init__(self, seq_len: int, n_features: int,
                 hidden_size: int = 32, n_layers: int = 3):
        super().__init__()
        input_dim = seq_len * n_features
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.Softsign()])
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_features)
        return self.net(x.reshape(x.size(0), -1))   # (batch, 1)
