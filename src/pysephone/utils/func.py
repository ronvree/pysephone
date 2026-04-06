import numpy as np


def round_partial(x: float, resolution: float) -> float:
    """Round `x` to the nearest multiple of `resolution`."""
    return round(x / resolution) * resolution

def create_left_mask(length: int, ix: int) -> np.ndarray:
    mask = np.arange(length)
    mask = np.where(mask >= ix, 1, 0)
    return mask