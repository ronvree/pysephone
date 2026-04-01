import torch


def batch_tensors(*ts):
    return torch.cat([t.unsqueeze(0) for t in ts], dim=0)



def create_left_mask(length: int, ixs: torch.Tensor) -> torch.Tensor:
    """
    Create tensors for masking the first `ixs` elements of a (batch of equal-length) time series
    :param length: length of time series
    :param ixs: torch tensor specifying the indices from which to mask the data
                tensor shape: (batch_size,)
    :return: torch tensor containing the masks
             tensor shape: (batch_size, length)
    """
    # Create an array of [0, 1, 2, 3, ... <season length>] and duplicate it for all samples in the batch
    mask = torch.arange(length, device=ixs.device).unsqueeze(0).expand(ixs.size(0), -1)
    # Create a mask by checking whether each entry is smaller than the sowing observation/index
    mask = (mask >= ixs.view(-1, 1)).to(torch.int)
    return mask


def create_right_mask(length: int, ixs: torch.Tensor) -> torch.Tensor:
    """
    Create tensors for masking the first `ixs` elements of a (batch of equal-length) time series
    :param length: length of time series
    :param ixs: torch tensor specifying the indices from which to mask the data
                tensor shape: (batch_size,)
    :return: torch tensor containing the masks
             tensor shape: (batch_size, length)
    """
    # Create an array of [0, 1, 2, 3, ... <season length>] and duplicate it for all samples in the batch
    mask = torch.arange(length, device=ixs.device).unsqueeze(0).expand(ixs.size(0), -1)
    # Create a mask by checking whether each entry is smaller than the sowing observation/index
    mask = (mask < ixs.view(-1, 1)).to(torch.int)
    return mask
