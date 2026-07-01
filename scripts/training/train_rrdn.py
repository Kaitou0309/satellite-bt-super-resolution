#!/usr/bin/env python3
import os
import random
import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bt_super_resolution.models import build_RRDN
from bt_super_resolution.normalization import load_normalization_stats

# Handle multi-GPU if available
strategy = tf.distribute.MirroredStrategy()

# ==================================================
# 1. Custom Loss / Metric Functions
# ==================================================

@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def spatial_gradient_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))


@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def total_physics_loss(y_true, y_pred):
    """
    Original value + gradient loss:
        value_loss = 0.5*MAE + 0.5*MSE
        total = value_loss + 0.2*gradient_loss
    Scaled by 100.
    """
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    value_loss = (0.5 * mae) + (0.5 * mse)

    alpha = 0.2
    grad_loss = spatial_gradient_loss(y_true, y_pred)

    total_loss = value_loss + (alpha * grad_loss)
    return total_loss * 100.0

# NOTE: This is the old version of the composite SSIM loss that was used in the initial experiments. 
# It is kept here for reference and comparison, 
# code below is used to compute loss to seen in log and check points.
@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def make_composite_ssim_loss(alpha=0.8):
    def composite_ssim_loss(y_true, y_pred):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))

        # the data is normalized and can have outliers, so we clip to a reasonable range before computing SSIM
        # [-3, 3] is a common range for z-score normalized data 
        # in the current data set, [-3, 3] covers 99.941% of values
        zmin, zmax = -3.0, 3.0
        # clip both y_true and y_pred to this range to prevent extreme outliers from dominating the SSIM calculation
        y_true_clip = tf.clip_by_value(y_true, zmin, zmax)
        y_pred_clip = tf.clip_by_value(y_pred, zmin, zmax)

        # tf.image.ssim works best when inputs are in the [0, 1] range, so we rescale our clipped tensors to that range
        # [0, 1] is the standard image-like range for SSIM 
        y_true_ssim = (y_true_clip - zmin) / (zmax - zmin)
        y_pred_ssim = (y_pred_clip - zmin) / (zmax - zmin)

        # compute SSIM score (1.0 is perfect similarity, -1.0 is worst)
        ssim_score = tf.image.ssim(y_true_ssim, y_pred_ssim, max_val=1.0)
        
        # 1 - (1 / batch_size) * sum(ssim_score) to convert from similarity to loss
        ssim_loss = 1.0 - tf.reduce_mean(ssim_score)

        # custom loss function 
        total_loss = ((1-alpha) * mae) + (alpha * ssim_loss)
        return total_loss * 100.0
    return composite_ssim_loss

@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def scaled_mae_metric(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mae * 100.0


@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def scaled_ssim_loss_metric(y_true, y_pred):
    zmin, zmax = -3.0, 3.0
    y_true_clip = tf.clip_by_value(y_true, zmin, zmax)
    y_pred_clip = tf.clip_by_value(y_pred, zmin, zmax)

    y_true_ssim = (y_true_clip - zmin) / (zmax - zmin)
    y_pred_ssim = (y_pred_clip - zmin) / (zmax - zmin)

    ssim_score = tf.image.ssim(y_true_ssim, y_pred_ssim, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim_score)
    return ssim_loss * 100.0


@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def charbonnier_loss(y_true, y_pred):
    epsilon = 1e-3
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(epsilon)))

# ==================================================
# 2. Parse arguments
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RRDN with Unified H5 Data & Dynamic Loss."
    )

    # -------- Model hyperparameters --------
    parser.add_argument("--n_rrdb", type=int, default=9, help="Number of RRDBs in RRDN.")
    parser.add_argument("--n_rdb_per_block", type=int, default=3, help="Number of RDBs inside each RRDB.")
    parser.add_argument("--n_conv_layers", type=int, default=5, help="Conv layers per RDB.")
    parser.add_argument("--growth_rate", type=int, default=64, help="Growth rate.")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels.")

    # -------- Training hyperparameters --------
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate.")

    # -------- Dynamic Loss Router --------
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="composite_ssim",
        choices=["mse", "mae", "charbonnier", "total_physics_loss", "composite_ssim"],
        help="Choose the loss function for training."
    )
    
    # -------- Composite SSIM specific --------
    parser.add_argument(
        "--ssim_alpha",
        type=float,
        default=0.8,
        help="Alpha weight for SSIM in composite loss (only used if loss_fn is composite_ssim)."
    )

    # -------- Optimizer Callbacks --------
    parser.add_argument("--lr_factor", type=float, default=0.5, help="LR reduction factor.")
    parser.add_argument("--lr_patience", type=int, default=4, help="Epochs to wait before reducing LR.")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Minimum LR boundary.")
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Patience for early stopping.")

    # -------- Data Paths --------
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training H5"
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        help="Optional evaluation H5. If omitted, 20 percent of training data is used."
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="metadata/unified_global_stats.json",
        help="Normalization statistics created for the training dataset."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints/rrdn",
        help="Directory for generated checkpoints."
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="",
        help="Optional: path to start from existing weights."
    )

    return parser.parse_args()


# ==================================================
# 3. Main Execution
# ==================================================

def main():
    args = parse_args()
    
    if args.loss_fn == "composite_ssim":
        loss_tag = f"{args.loss_fn}_alpha{args.ssim_alpha:.1f}"
    else: 
        loss_tag = args.loss_fn

    run_tag = (
        f"{args.n_rrdb}RRDB_"
        f"{args.n_rdb_per_block}RDB_"
        f"{args.n_conv_layers}convlayer_"
        f"g{args.growth_rate}_"
        f"UNI_{loss_tag.upper()}_"
        f"bs{args.batch_size}_"
        f"lr{args.lr:.0e}_"
        f"loss_fn{loss_tag}"
    )

    # Reproducibility
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("=" * 70)
    print(f"🚀 Training RRDN ({args.loss_fn.upper()} Phase) Tag: {run_tag}")
    print("=" * 70)

    # ------------------------------------------------
    # 4. Load Data
    # ------------------------------------------------
    print(f"🔄 Loading Training Data from {args.train_path}...")
    with h5py.File(args.train_path, "r") as f:
        X = f["L/bt"][:]
        Y = f["H/bt"][:]

    if args.eval_path:
        print(f"Loading evaluation data from {args.eval_path}...")
        with h5py.File(args.eval_path, "r") as f:
            X_val = f["L/bt"][:]
            Y_val = f["H/bt"][:]
        X_train, Y_train = X, Y
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=seed
        )

    scale_h = Y_train.shape[1] // X_train.shape[1]
    scale_w = Y_train.shape[2] // X_train.shape[2]
    print(f"📏 Detected Scaling Factor: {scale_h}x{scale_w}")

    stats = load_normalization_stats(args.stats_path)
    MU_X, SD_X = stats.mu_x, stats.sd_x
    MU_Y, SD_Y = stats.mu_y, stats.sd_y

    print("⚙️ Normalizing Data...")
    X_train = np.nan_to_num((X_train - MU_X) / (SD_X + 1e-8))
    Y_train = np.nan_to_num((Y_train - MU_Y) / (SD_Y + 1e-8))
    X_val = np.nan_to_num((X_val - MU_X) / (SD_X + 1e-8))
    Y_val = np.nan_to_num((Y_val - MU_Y) / (SD_Y + 1e-8))

    # Add channel dim if missing
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, -1)
    if len(Y_train.shape) == 3:
        Y_train = np.expand_dims(Y_train, -1)
    if len(X_val.shape) == 3:
        X_val = np.expand_dims(X_val, -1)
    if len(Y_val.shape) == 3:
        Y_val = np.expand_dims(Y_val, -1)

    # ------------------------------------------------
    # 5. Data pipeline
    # ------------------------------------------------
    def augment(lr_img, hr_img):
        if tf.random.uniform(()) > 0.5:
            lr_img = tf.image.flip_left_right(lr_img)
            hr_img = tf.image.flip_left_right(hr_img)
        if tf.random.uniform(()) > 0.5:
            lr_img = tf.image.flip_up_down(lr_img)
            hr_img = tf.image.flip_up_down(hr_img)
        return lr_img, hr_img

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------
    # 6. Loss router
    # ------------------------------------------------
    if args.loss_fn == "total_physics_loss":
        selected_loss = total_physics_loss
    elif args.loss_fn == "composite_ssim":
        selected_loss = make_composite_ssim_loss(alpha=args.ssim_alpha)
    elif args.loss_fn == "charbonnier":
        selected_loss = charbonnier_loss
    elif args.loss_fn == "mse":
        selected_loss = "mse"
    elif args.loss_fn == "mae":
        selected_loss = "mae"
    else:
        raise ValueError(f"Invalid loss function selected: {args.loss_fn}")

    # ------------------------------------------------
    # 7. Build model
    # ------------------------------------------------
    with strategy.scope():
        model = build_RRDN(
            scale_w=scale_w,
            scale_h=scale_h,
            n_rrdb=args.n_rrdb,
            n_rdb_per_block=args.n_rdb_per_block,
            n_conv_layers=args.n_conv_layers,
            growth_rate=args.growth_rate,
            channels=args.channels,
        )

        if args.pretrained_path and os.path.exists(args.pretrained_path):
            print(f"🧠 Injecting pre-trained weights from: {args.pretrained_path}")
            model.build((None, X_train.shape[1], X_train.shape[2], args.channels))
            model.load_weights(args.pretrained_path)

        model.compile(
            optimizer=Adam(learning_rate=args.lr, clipnorm=1.0),
            loss=selected_loss,
            metrics=[
                RootMeanSquaredError(name="rmse"),
                scaled_mae_metric,
                scaled_ssim_loss_metric,
            ],
        )

    # ------------------------------------------------
    # 8. Callbacks
    # ------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=1,
            min_lr=args.min_lr,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=os.path.join(args.checkpoint_dir, f"RRDN_{run_tag}_BEST.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # ------------------------------------------------
    # 9. Train
    # ------------------------------------------------
    print(f"🔥 Starting Training with {args.loss_fn.upper()} Loss...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    print(f"✅ Training complete! Best weights saved in {args.checkpoint_dir}")

    # Optional: quick summary of best validation loss
    best_val_loss = np.min(history.history["val_loss"])
    print(f"📉 Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
