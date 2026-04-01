import torch

# Simply wraps a list of tensors in a PyTorch dataset object
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)
