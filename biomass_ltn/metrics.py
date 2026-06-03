from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TARGET_INDEX, TARGET_NAMES
from .data import make_prediction_dataset, target_array
from .models import build_feature_extractor


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    split: str,
) -> pd.DataFrame:
    rows = [_metric_row(y_true, y_pred, mode, split, "ALL")]
    for idx, target in enumerate(TARGET_NAMES):
        rows.append(_metric_row(y_true[:, [idx]], y_pred[:, [idx]], mode, split, target))
    return pd.DataFrame(rows)


def rule_metrics(y_pred: np.ndarray, mode: str, split: str) -> pd.DataFrame:
    clover = y_pred[:, TARGET_INDEX["Dry_Clover_g"]]
    dead = y_pred[:, TARGET_INDEX["Dry_Dead_g"]]
    green = y_pred[:, TARGET_INDEX["Dry_Green_g"]]
    total = y_pred[:, TARGET_INDEX["Dry_Total_g"]]
    gdm = y_pred[:, TARGET_INDEX["GDM_g"]]

    order_violations = np.stack(
        [
            np.maximum(clover - total, 0.0),
            np.maximum(dead - total, 0.0),
            np.maximum(green - total, 0.0),
            np.maximum(gdm - total, 0.0),
            np.maximum(clover - gdm, 0.0),
            np.maximum(green - gdm, 0.0),
        ],
        axis=1,
    )

    return pd.DataFrame(
        [
            {
                "mode": mode,
                "split": split,
                "total_sum_mae_g": float(np.mean(np.abs(total - (clover + dead + green)))),
                "gdm_sum_mae_g": float(np.mean(np.abs(gdm - (clover + green)))),
                "negative_violation_g": float(np.mean(np.maximum(-y_pred, 0.0))),
                "order_violation_g": float(np.mean(order_violations)),
            }
        ]
    )


def save_predictions(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    output_path: str | Path,
    mode: str,
) -> None:
    output = frame[["image_id", "Sampling_Date", "State", "Species"]].copy()
    output["mode"] = mode
    for idx, target in enumerate(TARGET_NAMES):
        output[f"true_{target}"] = frame[target].to_numpy(dtype="float32")
        output[f"pred_{target}"] = y_pred[:, idx]
    output.to_csv(output_path, index=False)


def symbolic_baseline(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> np.ndarray:
    """Metadata-only baseline that predicts primitive masses and derives logical targets."""
    primitive_cols = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
    global_mean = train_frame[primitive_cols].mean()
    by_state_species = train_frame.groupby(["State", "Species"])[primitive_cols].mean()
    by_species = train_frame.groupby("Species")[primitive_cols].mean()
    by_state = train_frame.groupby("State")[primitive_cols].mean()

    rows = []
    for _, row in eval_frame.iterrows():
        key = (row["State"], row["Species"])
        if key in by_state_species.index:
            primitive = by_state_species.loc[key]
        elif row["Species"] in by_species.index:
            primitive = by_species.loc[row["Species"]]
        elif row["State"] in by_state.index:
            primitive = by_state.loc[row["State"]]
        else:
            primitive = global_mean

        clover, dead, green = primitive.to_numpy(dtype="float32")
        total = clover + dead + green
        gdm = clover + green
        rows.append([clover, dead, green, total, gdm])
    return np.asarray(rows, dtype="float32")


def representation_quality(
    model,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    image_size: int,
    batch_size: int,
    cache: bool,
) -> dict[str, float]:
    """Measure whether the learned shared representation is linearly predictive."""
    extractor = build_feature_extractor(model)
    train_inputs = make_prediction_dataset(
        train_frame,
        feature_columns,
        image_size=image_size,
        batch_size=batch_size,
        cache=cache,
    )
    test_inputs = make_prediction_dataset(
        test_frame,
        feature_columns,
        image_size=image_size,
        batch_size=batch_size,
        cache=cache,
    )

    z_train = extractor.predict(train_inputs, verbose=0)
    z_test = extractor.predict(test_inputs, verbose=0)
    y_train = target_array(train_frame)
    y_test = target_array(test_frame)

    if len(train_frame) < 3 or len(test_frame) < 2:
        return {"linear_probe_rmse": float("nan"), "linear_probe_r2": float("nan")}

    probe = Ridge(alpha=1.0)
    probe.fit(z_train, y_train)
    pred = probe.predict(z_test)
    return {
        "linear_probe_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "linear_probe_r2": float(r2_score(y_test, pred, multioutput="uniform_average")),
    }


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    split: str,
    target: str,
) -> dict[str, float | str]:
    return {
        "mode": mode,
        "split": split,
        "target": target,
        "mae_g": float(mean_absolute_error(y_true, y_pred)),
        "rmse_g": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }

