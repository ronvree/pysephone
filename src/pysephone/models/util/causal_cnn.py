import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        # x: (batch, channels, time)
        x = F.pad(x, (self.padding, 0))  # left padding only
        return self.conv(x)
    
    