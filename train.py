from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from biomass_ltn.config import TARGET_NAMES
from biomass_ltn.data import (
    make_dataset,
    make_prediction_dataset,
    prepare_data,
    target_array,
)
from biomass_ltn.logic import BiomassLTNObjective, target_tolerances_from_frame
from biomass_ltn.metrics import (
    regression_metrics,
    representation_quality,
    rule_metrics,
    save_predictions,
    symbolic_baseline,
)
from biomass_ltn.models import build_multimodal_regressor
from biomass_ltn.plots import generate_all_plots
from biomass_ltn.report import write_run_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train biomass regressors with Logical Tensor Networks.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=["ltn", "neural", "symbolic", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ltn-weight", type=float, default=0.10)
    parser.add_argument("--rule-tolerance", type=float, default=0.05)
    parser.add_argument("--ltn-warmup-epochs", type=int, default=5)
    parser.add_argument("--ltn-ramp-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--backbone", choices=["small_cnn", "mobilenetv2"], default="small_cnn")
    parser.add_argument("--mobilenet-weights", choices=["none", "imagenet"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)

    data = prepare_data(
        args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
        limit_samples=args.limit_samples,
    )
    save_config(args, data, run_dir)

    metric_frames: list[pd.DataFrame] = []
    rule_frames: list[pd.DataFrame] = []
    representation_rows: list[dict] = []
    history_frames: dict[str, pd.DataFrame] = {}
    prediction_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    trained_models: dict[str, tf.keras.Model] = {}

    modes = ["symbolic", "neural", "ltn"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"\n=== {mode.upper()} ===")
        if mode == "symbolic":
            y_pred = symbolic_baseline(data.train, data.test)
            y_true = target_array(data.test)
            prediction_data[mode] = (y_true, y_pred)
            collect_outputs(mode, data.test, y_pred, run_dir, metric_frames, rule_frames)
            continue

        model = build_multimodal_regressor(
            tabular_dim=data.tabular_dim,
            image_size=args.image_size,
            dropout=args.dropout,
            backbone=args.backbone,
            mobilenet_weights=args.mobilenet_weights,
            output_bias=normalized_target_logits(data).tolist(),
        )

        if mode == "neural":
            history = train_neural(model, data, args)
        else:
            history = train_ltn(model, data, args)

        history_frames[mode] = pd.DataFrame(history)
        history_frames[mode].to_csv(run_dir / f"training_history_{mode}.csv", index=False)

        # Save metrics first (in case model.save fails)
        y_pred = predict_raw(model, data.test, data, args)
        y_true = target_array(data.test)
        prediction_data[mode] = (y_true, y_pred)
        collect_outputs(mode, data.test, y_pred, run_dir, metric_frames, rule_frames)

        # Then save model
        try:
            model.save_weights(str(run_dir / "models" / f"{mode}_weights.h5"))
        except Exception as exc:
            print(f"  Warning: could not save {mode} weights: {exc}")

        trained_models[mode] = model

        probe = representation_quality(
            model,
            data.train,
            data.test,
            data.feature_columns,
            image_size=args.image_size,
            batch_size=args.batch_size,
            cache=args.cache_images,
        )
        probe["mode"] = mode
        representation_rows.append(probe)

    write_tables(run_dir, metric_frames, rule_frames, representation_rows)

    # Generate plots
    print("\nGenerating plots ...")
    generate_all_plots(
        run_dir=run_dir,
        history_frames=history_frames,
        metric_frames=metric_frames,
        rule_frames=rule_frames,
        prediction_data=prediction_data,
        models=trained_models,
        test_frame=data.test,
        feature_columns=data.feature_columns,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cache=args.cache_images,
    )

    report_path = write_run_report(
        run_dir,
        vars(args) | {"image_size": args.image_size, "batch_size": args.batch_size},
        metric_frames,
        rule_frames,
        representation_rows,
        history_frames,
    )
    print(f"\nSaved run artifacts to {run_dir}")
    print(f"Report: {report_path}")


# ---------------------------------------------------------------------------
# Neural training with history logging
# ---------------------------------------------------------------------------

def train_neural(model: tf.keras.Model, data, args: argparse.Namespace) -> list[dict]:
    train_ds = make_dataset(
        data.train,
        data.feature_columns,
        data.target_scale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augment,
        cache=args.cache_images,
        for_keras=True,
        seed=args.seed,
    )
    val_ds = make_dataset(
        data.val,
        data.feature_columns,
        data.target_scale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cache=args.cache_images,
        for_keras=True,
        seed=args.seed,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.Huber(delta=0.15),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ))

    keras_history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        callbacks=callbacks, verbose=2,
    )

    # Convert Keras history to our standard format
    history = []
    for epoch_idx in range(len(keras_history.history["loss"])):
        row = {
            "epoch": epoch_idx + 1,
            "loss": keras_history.history["loss"][epoch_idx],
            "rmse": keras_history.history["rmse"][epoch_idx],
            "mae": keras_history.history["mae"][epoch_idx],
            "val_loss": keras_history.history["val_loss"][epoch_idx],
            "val_rmse": keras_history.history["val_rmse"][epoch_idx],
            "val_mae": keras_history.history["val_mae"][epoch_idx],
        }
        history.append(row)
    return history


# ---------------------------------------------------------------------------
# LTN training with history logging and early stopping
# ---------------------------------------------------------------------------

def train_ltn(model: tf.keras.Model, data, args: argparse.Namespace) -> list[dict]:
    train_ds = make_dataset(
        data.train,
        data.feature_columns,
        data.target_scale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augment,
        cache=args.cache_images,
        seed=args.seed,
    )
    val_ds = make_dataset(
        data.val,
        data.feature_columns,
        data.target_scale,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cache=args.cache_images,
        seed=args.seed,
    )

    tolerances = target_tolerances_from_frame(data.train, TARGET_NAMES, data.target_scale)
    objective = BiomassLTNObjective(
        model,
        target_tolerances=tolerances,
        rule_tolerance=args.rule_tolerance,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    best_val_rmse = float("inf")
    best_weights = None
    patience_counter = 0

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        ltn_w = effective_ltn_weight(epoch, args)

        train_stats = run_ltn_epoch(
            model, objective, optimizer, train_ds,
            ltn_weight=ltn_w, training=True,
        )
        val_stats = run_ltn_epoch(
            model, objective, optimizer, val_ds,
            ltn_weight=ltn_w, training=False,
        )
        print(
            f"epoch {epoch:03d} "
            f"ltn_w={ltn_w:.4f} "
            f"loss={train_stats['loss']:.4f} sat={train_stats['sat']:.4f} rmse={train_stats['rmse']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_sat={val_stats['sat']:.4f} val_rmse={val_stats['rmse']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "ltn_weight": ltn_w,
            "loss": train_stats["loss"],
            "sat": train_stats["sat"],
            "rmse": train_stats["rmse"],
            "val_loss": val_stats["loss"],
            "val_sat": val_stats["sat"],
            "val_rmse": val_stats["rmse"],
        })

        # Early stopping on val RMSE
        if args.patience > 0:
            if val_stats["rmse"] < best_val_rmse:
                best_val_rmse = val_stats["rmse"]
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (best val_rmse={best_val_rmse:.4f})")
                    break

    # Restore best weights
    if best_weights is not None and args.patience > 0:
        model.set_weights(best_weights)
        print(f"  Restored best weights (val_rmse={best_val_rmse:.4f})")

    return history


def run_ltn_epoch(
    model: tf.keras.Model,
    objective: BiomassLTNObjective,
    optimizer: tf.keras.optimizers.Optimizer,
    dataset: tf.data.Dataset,
    ltn_weight: float,
    training: bool,
) -> dict[str, float]:
    loss_metric = tf.keras.metrics.Mean()
    sat_metric = tf.keras.metrics.Mean()
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()
    huber = tf.keras.losses.Huber(delta=0.15)

    for images, tabular, y_true in dataset:
        if training:
            with tf.GradientTape() as tape:
                y_pred = model([images, tabular], training=True)
                reg_loss = tf.reduce_mean(huber(y_true, y_pred))
                sat = objective.axioms_from_predictions(y_pred, y_true)
                loss = reg_loss + ltn_weight * (1.0 - sat)
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients_and_vars = [
                (grad, var) for grad, var in zip(gradients, model.trainable_variables) if grad is not None
            ]
            optimizer.apply_gradients(gradients_and_vars)
        else:
            y_pred = model([images, tabular], training=False)
            reg_loss = tf.reduce_mean(huber(y_true, y_pred))
            sat = objective.axioms_from_predictions(y_pred, y_true)
            loss = reg_loss + ltn_weight * (1.0 - sat)

        loss_metric.update_state(loss)
        sat_metric.update_state(sat)
        rmse_metric.update_state(y_true, y_pred)

    return {
        "loss": float(loss_metric.result().numpy()),
        "sat": float(sat_metric.result().numpy()),
        "rmse": float(rmse_metric.result().numpy()),
    }


def predict_raw(model: tf.keras.Model, frame: pd.DataFrame, data, args: argparse.Namespace) -> np.ndarray:
    pred_ds = make_prediction_dataset(
        frame,
        data.feature_columns,
        image_size=args.image_size,
        batch_size=args.batch_size,
        cache=args.cache_images,
    )
    pred_norm = model.predict(pred_ds, verbose=0)
    return pred_norm.astype("float32") * np.float32(data.target_scale)


def normalized_target_logits(data) -> np.ndarray:
    means = data.train[TARGET_NAMES].to_numpy(dtype="float32").mean(axis=0) / np.float32(data.target_scale)
    means = np.clip(means, 1e-4, None)
    return np.log(np.exp(means) - 1.0).astype("float32")


def effective_ltn_weight(epoch: int, args: argparse.Namespace) -> float:
    if epoch <= args.ltn_warmup_epochs:
        return 0.0
    ramp_epochs = max(args.ltn_ramp_epochs, 1)
    ramp_position = min(epoch - args.ltn_warmup_epochs, ramp_epochs)
    return args.ltn_weight * (ramp_position / ramp_epochs)


def collect_outputs(
    mode: str,
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    run_dir: Path,
    metric_frames: list[pd.DataFrame],
    rule_frames: list[pd.DataFrame],
) -> None:
    y_true = target_array(frame)
    metric_frame = regression_metrics(y_true, y_pred, mode=mode, split="test")
    rule_frame = rule_metrics(y_pred, mode=mode, split="test")
    metric_frames.append(metric_frame)
    rule_frames.append(rule_frame)
    metric_frame.to_csv(run_dir / f"metrics_{mode}.csv", index=False)
    rule_frame.to_csv(run_dir / f"rules_{mode}.csv", index=False)
    save_predictions(frame, y_pred, run_dir / "predictions" / f"{mode}.csv", mode)
    print(metric_frame[metric_frame["target"] == "ALL"].to_string(index=False))
    print(rule_frame.to_string(index=False))


def write_tables(
    run_dir: Path,
    metric_frames: list[pd.DataFrame],
    rule_frames: list[pd.DataFrame],
    representation_rows: list[dict],
) -> None:
    if metric_frames:
        pd.concat(metric_frames, ignore_index=True).to_csv(run_dir / "metrics_all.csv", index=False)
    if rule_frames:
        pd.concat(rule_frames, ignore_index=True).to_csv(run_dir / "rules_all.csv", index=False)
    if representation_rows:
        pd.DataFrame(representation_rows).to_csv(run_dir / "representation_probe.csv", index=False)


def save_config(args: argparse.Namespace, data, run_dir: Path) -> None:
    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    config["target_scale"] = data.target_scale
    config["tabular_dim"] = data.tabular_dim
    config["n_train"] = len(data.train)
    config["n_val"] = len(data.val)
    config["n_test"] = len(data.test)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


if __name__ == "__main__":
    main()
