import torch
import torch.nn as nn


class ActuatorGRU(nn.Module):
    """Multi-layer GRU torque estimator (Zhu et al., 2023).

    Accepts a 3-D tensor (batch, seq_len, n_features), processes the sequence
    through `n_layers` stacked GRU cells, and maps the last hidden state of
    the final layer to a scalar torque prediction via a fully-connected layer.
    """

    def __init__(self, n_features: int, hidden_size: int = 64,
                 n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (batch, seq_len, n_features)
        # h_n : (n_layers, batch, hidden_size)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])   # (batch, 1)
