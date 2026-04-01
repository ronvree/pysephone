def round_partial(x: float, resolution: float) -> float:
    """Round `x` to the nearest multiple of `resolution`."""
    return round(x / resolution) * resolution
