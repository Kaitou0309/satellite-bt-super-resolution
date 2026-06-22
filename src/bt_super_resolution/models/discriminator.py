"""PatchGAN-style discriminator used during adversarial training."""

from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, LeakyReLU


def _discriminator_block(x, filters: int, stride: int, use_batchnorm: bool, name: str):
    x = Conv2D(
        filters,
        3,
        strides=stride,
        padding="same",
        use_bias=not use_batchnorm,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)
    if use_batchnorm:
        x = BatchNormalization(name=f"{name}_bn")(x)
    return LeakyReLU(negative_slope=0.2, name=f"{name}_lrelu")(x)


def build_patch_discriminator(*, channels: int = 1, use_batchnorm: bool = True) -> Model:
    inputs = Input(shape=(None, None, channels), name="d_input")
    x = Conv2D(
        64,
        3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        name="d_block1_conv",
    )(inputs)
    x = LeakyReLU(negative_slope=0.2, name="d_block1_lrelu")(x)
    block_specs = (
        (64, 2),
        (128, 1),
        (128, 2),
        (256, 1),
        (256, 2),
        (512, 1),
        (512, 2),
        (512, 1),
        (256, 1),
    )
    for block_index, (filters, stride) in enumerate(block_specs, start=2):
        x = _discriminator_block(x, filters, stride, use_batchnorm, f"d_block{block_index}")
    outputs = Conv2D(
        1,
        3,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        name="d_patch_prob",
    )(x)
    return Model(inputs, outputs, name="DeepPatchDiscriminator")
