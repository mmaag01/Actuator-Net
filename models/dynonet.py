import torch
import torch.nn as nn
from dynonet.lti import MimoLinearDynamicalOperator, StableSecondOrderMimoLinearDynamicalOperator


class WienerHammersteinNet(nn.Module):
    """Wiener-Hammerstein structure: G1 (linear IIR) -> MLP (static nonlinearity) -> G2 (linear IIR).

    When stable=True, G-blocks use the sigmoid-based 2nd-order pole reparametrization
    from the dynoNet paper, guaranteeing |poles| < 1.
    When stable=False, G-blocks are general IIR filters with learnable na/nb coefficients
    initialized from Uniform(-0.01, 0.01).

    dynoNet's IIR filtering is implemented via scipy.signal (CPU-only). _apply is
    overridden to prevent parameters from migrating to CUDA; forward moves inputs to
    CPU internally and returns output on the caller's original device.

    Input:  (batch, seq_len, n_features)
    Output: (batch, 1)  — prediction at the last timestep of each window
    """

    def __init__(self, n_features: int, n_channels: int = 8,
                 na: int = 2, nb: int = 2, mlp_hidden: int = 20,
                 stable: bool = True):
        super().__init__()

        if stable:
            self.G1 = StableSecondOrderMimoLinearDynamicalOperator(n_features, n_channels)
            self.G2 = StableSecondOrderMimoLinearDynamicalOperator(n_channels, 1)
        else:
            self.G1 = MimoLinearDynamicalOperator(n_features, n_channels, n_b=nb, n_a=na)
            self.G2 = MimoLinearDynamicalOperator(n_channels, 1, n_b=nb, n_a=na)
            # MimoLinearDynamicalOperator already initialises with Uniform(-0.01, 0.01)

        self.mlp = nn.Sequential(
            nn.Linear(n_channels, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, n_channels),
        )

    # dynoNet calls scipy.signal internally — it cannot run on CUDA.
    # Intercept .to() / .cuda() so parameters stay on CPU.
    def _apply(self, fn, recurse=True):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) — any device
        device = x.device
        x_cpu = x.cpu()

        y1 = self.G1(x_cpu)                                               # (batch, seq_len, n_channels), CPU
        batch, seq_len, ch = y1.shape
        y2 = self.mlp(y1.reshape(-1, ch)).reshape(batch, seq_len, ch)     # (batch, seq_len, n_channels), CPU
        y3 = self.G2(y2)                                                   # (batch, seq_len, 1), CPU

        return y3[:, -1, :].to(device)                                    # (batch, 1), original device
