import torch
import torch.nn as nn
import torch.nn.functional as F


# class TTCNN(nn.Module):
#     """
#         Thermal time CNN
#     """
#
#     def __init__(self,
#                  num_channels_in: int,
#                  num_channels_out: int,
#                  hidden_size: int,
#                  kernel_size: int,
#                  num_layers: int = 3,
#                  final_activation: nn.Module = None,
#                  ):
#         super().__init__()
#         assert num_channels_in >= 1
#         assert num_channels_out >= 1
#         assert num_layers >= 1
#         assert hidden_size >= 1
#         if final_activation is None:
#             final_activation = nn.Identity()
#
#         stride = 1  # Fixed
#
#         if num_layers == 1:
#
#             self._cnn = nn.Conv1d(
#                 num_channels_in,
#                 num_channels_out,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=kernel_size // 2,
#             )
#
#         else:
#
#             l_in = nn.Conv1d(
#                 num_channels_in,
#                 hidden_size,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=kernel_size // 2,
#             )
#
#             l_out = nn.Conv1d(
#                 hidden_size,
#                 num_channels_out,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=kernel_size // 2,
#             )
#
#             self._cnn = nn.Sequential(*(
#                 [l_in, nn.ReLU(),] +
#                 [nn.Sequential(*[nn.Conv1d(
#                     in_channels=hidden_size,
#                     out_channels=hidden_size,
#                     kernel_size=kernel_size,
#                     stride=stride,
#                     padding=kernel_size // 2,
#                 ), nn.ReLU()]) for _ in range(num_layers - 2)] +
#                 [l_out, final_activation]
#             ))
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # input should be of shape (B, C_in, L), where
#         #   B is batch size
#         #   C_in is nr of input channels
#         #   L is sequence length
#
#         # output will be of shape (B, C_out, L), where
#         #   C_out is nr of output channels
#         return self._cnn(x)


class TorchGDD(nn.Module):

    def __init__(self,
                 t_base: float,
                 t_base_req_grad: bool = True,
                 dtype=None,
                 device=None,
                 ):
        super().__init__()

        parameter_kwargs = {}
        if dtype is not None:
            parameter_kwargs['dtype'] = dtype
        if device is not None:
            parameter_kwargs['device'] = device

        tb = torch.tensor(float(t_base), **parameter_kwargs)

        self._tb = nn.Parameter(tb, requires_grad=t_base_req_grad)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        return self.f_gdd(ts, self._tb)

    @staticmethod
    def f_gdd(ts: torch.Tensor, tb: torch.Tensor):
        return F.relu(ts - tb)
