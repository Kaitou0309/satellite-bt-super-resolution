"""Public adapter for the original Chen RRDN architecture implementation."""

from tensorflow.keras import Model

from .rrdn_chen import build_RRDN


def build_rrdn(
    *,
    scale: int = 4,
    channels: int = 1,
    num_rrdb_blocks: int = 9,
    rdb_per_rrdb: int = 3,
    conv_layers_per_rdb: int = 5,
    filters: int = 64,
    growth_rate: int = 64,
    residual_scaling: float = 0.2,
    upsampling: str = "bilinear",
) -> Model:
    """Map public config names to the original ``build_RRDN`` signature."""

    if filters != growth_rate:
        raise ValueError(
            "The original RRDN architecture uses growth_rate for both feature "
            f"width and dense growth; received filters={filters}, growth_rate={growth_rate}."
        )
    if residual_scaling != 0.2:
        raise ValueError("The original RRDN architecture fixes residual scaling at 0.2.")
    if upsampling.lower() != "bilinear":
        raise ValueError("The original release architecture uses bilinear upsampling.")

    return build_RRDN(
        scale_w=scale,
        scale_h=scale,
        n_rrdb=num_rrdb_blocks,
        n_rdb_per_block=rdb_per_rrdb,
        n_conv_layers=conv_layers_per_rdb,
        growth_rate=growth_rate,
        channels=channels,
    )
