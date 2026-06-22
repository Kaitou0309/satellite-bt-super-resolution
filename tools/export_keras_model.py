"""Export a released weights-only RRDN generator as a portable Keras model."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import tensorflow as tf

from bt_super_resolution import load_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=None,
        help="Optionally rewrite a legacy HDF5 checkpoint in current .weights.h5 format.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    bundle = load_generator(args.config, weights_path=args.weights)

    def clone_layer(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        return layer.__class__.from_config(layer.get_config())

    exported = tf.keras.models.clone_model(bundle.model, clone_function=clone_layer)
    exported.set_weights(bundle.model.get_weights())

    test_input = tf.random.stateless_normal((1, 8, 8, 1), seed=(137, 42))
    source_output = bundle.model(test_input, training=False).numpy()
    export_output = exported(test_input, training=False).numpy()
    np.testing.assert_allclose(source_output, export_output, rtol=1e-5, atol=1e-6)

    if args.weights_output is not None:
        args.weights_output.parent.mkdir(parents=True, exist_ok=True)
        exported.save_weights(args.weights_output)
        weights_check = tf.keras.models.clone_model(exported)
        weights_check.load_weights(args.weights_output)
        weights_output = weights_check(test_input, training=False).numpy()
        np.testing.assert_allclose(source_output, weights_output, rtol=1e-5, atol=1e-6)
        print(f"Weights: {args.weights_output}")
        print(f"Weights SHA-256: {file_sha256(args.weights_output)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported.save(args.output)

    restored = tf.keras.models.load_model(args.output, compile=False)
    restored_output = restored(test_input, training=False).numpy()
    np.testing.assert_allclose(source_output, restored_output, rtol=1e-5, atol=1e-6)

    print(f"Exported: {args.output}")
    print(f"SHA-256: {file_sha256(args.output)}")
    print(f"Verified output shape: {restored_output.shape}")


if __name__ == "__main__":
    main()
