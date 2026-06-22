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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bt_super_resolution.models import build_patch_discriminator, build_RRDN


# ==================================================
# Multi-GPU Strategy
# ==================================================
USE_DISTRIBUTED = os.environ.get("USE_DISTRIBUTED", "1") == "1"

if USE_DISTRIBUTED:
    strategy = tf.distribute.MirroredStrategy()
    print("Using MirroredStrategy")
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
    print("Using OneDeviceStrategy on CPU")


# ==================================================
# 1. Custom Loss / Metric Functions
# ==================================================

@tf.keras.utils.register_keras_serializable(package="CustomPhysics")
def spatial_gradient_loss(y_true, y_pred):
    grad_true = tf.image.sobel_edges(y_true)
    grad_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(grad_true - grad_pred))


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
def composite_ssim_loss(y_true, y_pred, alpha=0.8):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    zmin, zmax = -3.0, 3.0
    y_true_clip = tf.clip_by_value(y_true, zmin, zmax)
    y_pred_clip = tf.clip_by_value(y_pred, zmin, zmax)

    y_true_ssim = (y_true_clip - zmin) / (zmax - zmin)
    y_pred_ssim = (y_pred_clip - zmin) / (zmax - zmin)

    ssim_score = tf.image.ssim(y_true_ssim, y_pred_ssim, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim_score)

    total_loss = ((1.0 - alpha) * mae) + (alpha * ssim_loss)
    return total_loss * 100.0


# ==================================================
# 3. Argument Parser
# ==================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained RRDN generator with PatchGAN discriminator."
    )

    # -------- Generator architecture --------
    parser.add_argument("--n_rrdb", type=int, default=9)
    parser.add_argument("--n_rdb_per_block", type=int, default=3)
    parser.add_argument("--n_conv_layers", type=int, default=5)
    parser.add_argument("--growth_rate", type=int, default=64)
    parser.add_argument("--channels", type=int, default=1)

    # -------- Training hyperparameters --------
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--g_lr", type=float, default=1e-5)
    parser.add_argument("--d_lr", type=float, default=1e-5)

    # -------- GAN loss weights --------
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1e-4)
    parser.add_argument("--lambda_grad", type=float, default=0.0)
    parser.add_argument("--ssim_alpha", type=float, default=0.8)

    # -------- Data paths --------
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to paired training HDF5 data."
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        default="metadata/unified_global_stats.npz",
        help="Normalization statistics created for the training dataset."
    )

    # -------- Pretrained generator weights --------
    parser.add_argument(
        "--pretrained_generator_path",
        type=str,
        required=True,
        help="Path to pretrained RRDN generator weights .h5 file."
    )

    # -------- Output --------
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints/gan"
    )

    parser.add_argument(
        "--use_batchnorm_d",
        action="store_true",
        help="Use BatchNorm inside discriminator blocks."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a lightweight compile/sanity test using only a few samples and one train/val batch."
    )

    return parser.parse_args()


# ==================================================
# 4. Main
# ==================================================

def main():
    args = parse_args()

    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_tag = (
        f"RRDN_GAN_"
        f"{args.n_rrdb}RRDB_"
        f"{args.n_rdb_per_block}RDB_"
        f"{args.n_conv_layers}conv_"
        f"g{args.growth_rate}_"
        f"bs{args.batch_size}_"
        f"glr{args.g_lr:.0e}_"
        f"dlr{args.d_lr:.0e}_"
        f"adv{args.lambda_adv:.0e}"
    )

    print("=" * 80)
    print(f"🚀 Starting RRDN-GAN fine-tuning: {run_tag}")
    print("=" * 80)

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    print(f"🔄 Loading training data from: {args.train_path}")

    with h5py.File(args.train_path, "r") as f:
        X = f["L/bt"][:]
        Y = f["H/bt"][:]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    scale_h = Y_train.shape[1] // X_train.shape[1]
    scale_w = Y_train.shape[2] // X_train.shape[2]

    print(f"📏 Detected scale_h={scale_h}, scale_w={scale_w}")

    # ------------------------------------------------
    # Normalize data
    # ------------------------------------------------
    if not os.path.isfile(args.stats_path):
        raise FileNotFoundError(f"Normalization statistics not found: {args.stats_path}")
    with np.load(args.stats_path) as s:
        MU_X, SD_X = float(s["mu_X"]), float(s["sd_X"])
        MU_Y, SD_Y = float(s["mu_Y"]), float(s["sd_Y"])
    print(f"📊 Loaded normalization stats from {args.stats_path}")

    print("⚙️ Normalizing data...")

    X_train = np.nan_to_num((X_train - MU_X) / (SD_X + 1e-8))
    Y_train = np.nan_to_num((Y_train - MU_Y) / (SD_Y + 1e-8))
    X_val = np.nan_to_num((X_val - MU_X) / (SD_X + 1e-8))
    Y_val = np.nan_to_num((Y_val - MU_Y) / (SD_Y + 1e-8))

    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, -1)
    if len(Y_train.shape) == 3:
        Y_train = np.expand_dims(Y_train, -1)
    if len(X_val.shape) == 3:
        X_val = np.expand_dims(X_val, -1)
    if len(Y_val.shape) == 3:
        Y_val = np.expand_dims(Y_val, -1)
        
    if args.dry_run:
        print("🧪 DRY RUN MODE ENABLED")
        print("Using only a few samples and forcing epochs=1.")

        n_dry = max(2, args.batch_size * 2)

        X_train = X_train[:n_dry]
        Y_train = Y_train[:n_dry]
        X_val = X_val[:n_dry]
        Y_val = Y_val[:n_dry]

        args.epochs = 1

    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"Y_val shape:   {Y_val.shape}")

    # ------------------------------------------------
    # Data pipeline
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

    if args.dry_run:
        train_ds = train_ds.map(augment, num_parallel_calls=1)
        train_ds = train_ds.shuffle(16).batch(args.batch_size).prefetch(1)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_ds = val_ds.batch(args.batch_size).prefetch(1)

        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1

        train_ds = train_ds.with_options(options)
        val_ds = val_ds.with_options(options)

    else:
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    if args.dry_run:
        train_dist_ds = train_ds
        val_dist_ds = val_ds
    else:
        train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
        val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    # ------------------------------------------------
    # Build generator + discriminator
    # ------------------------------------------------
    with strategy.scope():
        generator = build_RRDN(
            scale_w=scale_w,
            scale_h=scale_h,
            n_rrdb=args.n_rrdb,
            n_rdb_per_block=args.n_rdb_per_block,
            n_conv_layers=args.n_conv_layers,
            growth_rate=args.growth_rate,
            channels=args.channels,
        )

        generator.build((None, X_train.shape[1], X_train.shape[2], args.channels))

        if not os.path.exists(args.pretrained_generator_path):
            raise FileNotFoundError(
                f"Pretrained generator weights not found: {args.pretrained_generator_path}"
            )

        print(f"🧠 Loading pretrained generator weights from: {args.pretrained_generator_path}")
        generator.load_weights(args.pretrained_generator_path)

        discriminator = build_patch_discriminator(
            channels=args.channels,
            use_batchnorm=args.use_batchnorm_d
        )

        g_optimizer = Adam(learning_rate=args.g_lr, beta_1=0.5, beta_2=0.999, clipnorm=1.0)
        d_optimizer = Adam(learning_rate=args.d_lr, beta_1=0.5, beta_2=0.999, clipnorm=1.0)

        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        )

        train_rmse = RootMeanSquaredError(name="train_rmse")
        val_rmse = RootMeanSquaredError(name="val_rmse")

    # ------------------------------------------------
    # Loss functions
    # ------------------------------------------------
    def discriminator_loss(real_prob, fake_prob):
        real_loss = bce(tf.ones_like(real_prob), real_prob)
        fake_loss = bce(tf.zeros_like(fake_prob), fake_prob)

        real_loss = tf.reduce_mean(real_loss)
        fake_loss = tf.reduce_mean(fake_loss)

        return real_loss + fake_loss

    def generator_adv_loss(fake_prob):
        adv_loss = bce(tf.ones_like(fake_prob), fake_prob)
        return tf.reduce_mean(adv_loss)

    def reconstruction_loss(hr_bt, sr_bt):
        return composite_ssim_loss(hr_bt, sr_bt, alpha=args.ssim_alpha)

    # ------------------------------------------------
    # Train step
    # ------------------------------------------------
    def step_fn(lr_bt, hr_bt):
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            sr_bt = generator(lr_bt, training=True)

            real_prob = discriminator(hr_bt, training=True)
            fake_prob_for_d = discriminator(tf.stop_gradient(sr_bt), training=True)
            fake_prob_for_g = discriminator(sr_bt, training=True)

            d_loss = discriminator_loss(real_prob, fake_prob_for_d)

            rec_loss = reconstruction_loss(hr_bt, sr_bt)
            adv_loss = generator_adv_loss(fake_prob_for_g)
            grad_loss = spatial_gradient_loss(hr_bt, sr_bt)

            g_loss = (
                args.lambda_rec * rec_loss
                + args.lambda_adv * adv_loss
                + args.lambda_grad * grad_loss
            )

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)

        d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        train_rmse.update_state(hr_bt, sr_bt)

        return d_loss, g_loss, rec_loss, adv_loss, grad_loss

    @tf.function
    def distributed_train_step(lr_bt, hr_bt):
        per_replica_losses = strategy.run(step_fn, args=(lr_bt, hr_bt))

        d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[0], axis=None)
        g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[1], axis=None)
        rec_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[2], axis=None)
        adv_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[3], axis=None)
        grad_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses[4], axis=None)

        return d_loss, g_loss, rec_loss, adv_loss, grad_loss

    # ------------------------------------------------
    # Validation step
    # ------------------------------------------------
    def val_step_fn(lr_bt, hr_bt):
        sr_bt = generator(lr_bt, training=False)

        rec_loss = reconstruction_loss(hr_bt, sr_bt)
        mae = tf.reduce_mean(tf.abs(hr_bt - sr_bt))
        ssim_loss = scaled_ssim_loss_metric(hr_bt, sr_bt)

        val_rmse.update_state(hr_bt, sr_bt)

        return rec_loss, mae, ssim_loss

    @tf.function
    def distributed_val_step(lr_bt, hr_bt):
        per_replica_vals = strategy.run(val_step_fn, args=(lr_bt, hr_bt))

        rec_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_vals[0], axis=None)
        mae = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_vals[1], axis=None)
        ssim_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_vals[2], axis=None)

        return rec_loss, mae, ssim_loss

    # ------------------------------------------------
    # Checkpoints
    # ------------------------------------------------
    best_val_rec = np.inf

    best_generator_path = os.path.join(args.checkpoint_dir, f"{run_tag}_GENERATOR_BEST.weights.h5")
    final_generator_path = os.path.join(args.checkpoint_dir, f"{run_tag}_GENERATOR_FINAL.weights.h5")
    final_discriminator_path = os.path.join(args.checkpoint_dir, f"{run_tag}_DISCRIMINATOR_FINAL.weights.h5")

    # ------------------------------------------------
    # Training loop
    # ------------------------------------------------
    print("🔥 Starting GAN fine-tuning...")

    for epoch in range(1, args.epochs + 1):
        train_rmse.reset_state()
        val_rmse.reset_state()

        train_d_losses = []
        train_g_losses = []
        train_rec_losses = []
        train_adv_losses = []
        train_grad_losses = []

        for step, (lr_bt, hr_bt) in enumerate(train_dist_ds):
            d_loss, g_loss, rec_loss, adv_loss, grad_loss = distributed_train_step(lr_bt, hr_bt)

            train_d_losses.append(float(d_loss))
            train_g_losses.append(float(g_loss))
            train_rec_losses.append(float(rec_loss))
            train_adv_losses.append(float(adv_loss))
            train_grad_losses.append(float(grad_loss))
            
            if args.dry_run:
                print("🧪 DRY RUN: completed one training batch.")
                break

        val_rec_losses = []
        val_maes = []
        val_ssims = []

        for step, (lr_bt, hr_bt) in enumerate(val_dist_ds):
            val_rec, val_mae, val_ssim = distributed_val_step(lr_bt, hr_bt)

            val_rec_losses.append(float(val_rec))
            val_maes.append(float(val_mae))
            val_ssims.append(float(val_ssim))

            if args.dry_run:
                print("🧪 DRY RUN: completed one validation batch.")
                break

        epoch_train_d = np.mean(train_d_losses)
        epoch_train_g = np.mean(train_g_losses)
        epoch_train_rec = np.mean(train_rec_losses)
        epoch_train_adv = np.mean(train_adv_losses)
        epoch_train_grad = np.mean(train_grad_losses)

        epoch_val_rec = np.mean(val_rec_losses)
        epoch_val_mae = np.mean(val_maes)
        epoch_val_ssim_loss = np.mean(val_ssims)

        epoch_train_rmse = float(train_rmse.result())
        epoch_val_rmse = float(val_rmse.result())

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"D_loss={epoch_train_d:.4f} | "
            f"G_loss={epoch_train_g:.4f} | "
            f"rec={epoch_train_rec:.4f} | "
            f"adv={epoch_train_adv:.4f} | "
            f"grad={epoch_train_grad:.4f} | "
            f"train_rmse={epoch_train_rmse:.4f} | "
            f"val_rec={epoch_val_rec:.4f} | "
            f"val_mae={epoch_val_mae:.4f} | "
            f"val_rmse={epoch_val_rmse:.4f} | "
            f"val_ssim_loss={epoch_val_ssim_loss:.4f}"
        )

        if epoch_val_rec < best_val_rec:
            best_val_rec = epoch_val_rec
            generator.save_weights(best_generator_path)
            print(f"✅ New best generator saved: {best_generator_path}")

    generator.save_weights(final_generator_path)
    discriminator.save_weights(final_discriminator_path)

    print("=" * 80)
    print("✅ GAN fine-tuning complete.")
    print(f"Best generator:       {best_generator_path}")
    print(f"Final generator:      {final_generator_path}")
    print(f"Final discriminator:  {final_discriminator_path}")
    print(f"Best val_rec:         {best_val_rec:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
