import tensorflow as tf

from bt_super_resolution.models import build_patch_discriminator, build_rrdn


def test_rrdn_output_is_four_times_larger() -> None:
    model = build_rrdn(
        scale=4,
        channels=1,
        num_rrdb_blocks=1,
        rdb_per_rrdb=1,
        conv_layers_per_rdb=1,
        filters=8,
        growth_rate=8,
    )
    output = model(tf.zeros((2, 8, 10, 1)), training=False)
    assert tuple(output.shape) == (2, 32, 40, 1)


def test_patch_discriminator_returns_patch_map() -> None:
    model = build_patch_discriminator(channels=1, use_batchnorm=True)
    output = model(tf.zeros((2, 64, 64, 1)), training=False)
    assert tuple(output.shape) == (2, 4, 4, 1)
