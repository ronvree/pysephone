import torch
import torch.nn as nn

from pysephone.models.util.causal_cnn import CausalConv1d


class TTCNN(nn.Module):
    """
    Thermal time CNN (causal).
    Input:  (B, C_in, L) temperature features over time
    Output: (B, C_out, L) phenological contribution in [0, 1] per timestep
    """

    def __init__(
        self,
        num_channels_in: int,
        num_channels_out: int,
        hidden_size: int,
        kernel_size: int,
        num_layers: int = 3,
        final_activation: nn.Module | None = None,
        dilation: int = 1,
        use_dilations: bool = False,
    ):
        super().__init__()
        assert num_channels_in >= 1
        assert num_channels_out >= 1
        assert num_layers >= 1
        assert hidden_size >= 1
        assert kernel_size >= 1

        # Output between 0-1
        if final_activation is None:
            final_activation = nn.Sigmoid()

        layers: list[nn.Module] = []

        # Choose dilation per layer (optional)
        def d_for_layer(i: int) -> int:
            if not use_dilations:
                return dilation
            # common TCN-style schedule: 1, 2, 4, 8, ...
            return dilation * (2 ** i)

        if num_layers == 1:
            layers.append(CausalConv1d(num_channels_in, num_channels_out, kernel_size, dilation=d_for_layer(0)))
            layers.append(final_activation)
        else:
            # input block
            layers.append(CausalConv1d(num_channels_in, hidden_size, kernel_size, dilation=d_for_layer(0)))
            layers.append(nn.ReLU())

            # hidden blocks
            for i in range(1, num_layers - 1):
                layers.append(CausalConv1d(hidden_size, hidden_size, kernel_size, dilation=d_for_layer(i)))
                layers.append(nn.ReLU())

            # output block
            layers.append(CausalConv1d(hidden_size, num_channels_out, kernel_size, dilation=d_for_layer(num_layers - 1)))
            layers.append(final_activation)

        self._cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._cnn(x)
