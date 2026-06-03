"""Post-training visualizations for biomass LTN runs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .config import TARGET_NAMES
from .data import make_prediction_dataset, target_array
from .models import build_feature_extractor

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
_PALETTE = {"symbolic": "#636EFA", "neural": "#EF553B", "ltn": "#00CC96"}

def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05, rc={
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history_frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Plot loss, RMSE, and (for LTN) satisfiability curves."""
    _setup_style()
    output_dir = Path(output_dir)

    # ---- Loss ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, df in history_frames.items():
        color = _PALETTE.get(mode, None)
        ax.plot(df["epoch"], df["loss"], label=f"{mode} train", color=color)
        if "val_loss" in df.columns:
            ax.plot(df["epoch"], df["val_loss"], "--", label=f"{mode} val",
                    color=color, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(output_dir / "loss_curves.png")
    plt.close(fig)

    # ---- RMSE ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, df in history_frames.items():
        color = _PALETTE.get(mode, None)
        ax.plot(df["epoch"], df["rmse"], label=f"{mode} train", color=color)
        if "val_rmse" in df.columns:
            ax.plot(df["epoch"], df["val_rmse"], "--", label=f"{mode} val",
                    color=color, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("Training & Validation RMSE")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(output_dir / "rmse_curves.png")
    plt.close(fig)

    # ---- Satisfiability (LTN only) ----
    ltn_dfs = {m: df for m, df in history_frames.items() if "sat" in df.columns}
    if ltn_dfs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, df in ltn_dfs.items():
            color = _PALETTE.get(mode, None)
            ax.plot(df["epoch"], df["sat"], label=f"{mode} train", color=color)
            if "val_sat" in df.columns:
                ax.plot(df["epoch"], df["val_sat"], "--", label=f"{mode} val",
                        color=color, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Satisfiability")
        ax.set_title("LTN Axiom Satisfiability Over Training")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fontsize=8)
        fig.savefig(output_dir / "satisfiability_curve.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Predicted vs Actual scatter
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str,
    output_dir: str | Path,
) -> None:
    """Per-target predicted vs actual scatter with identity line."""
    _setup_style()
    output_dir = Path(output_dir)
    n_targets = len(TARGET_NAMES)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4), squeeze=False)

    for idx, (name, ax) in enumerate(zip(TARGET_NAMES, axes[0])):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        ax.scatter(true_col, pred_col, alpha=0.5, s=18,
                   color=_PALETTE.get(mode, "#636EFA"), edgecolors="none")
        lims = [0, max(true_col.max(), pred_col.max()) * 1.1 + 1]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual (g)")
        ax.set_ylabel("Predicted (g)")
        ax.set_title(name.replace("_", " "), fontsize=10)

    fig.suptitle(f"Predicted vs Actual — {mode.upper()}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_vs_actual_{mode}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rule violation comparison
# ---------------------------------------------------------------------------

def plot_rule_violations(
    rule_frames: list[pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Bar chart comparing rule violations across modes."""
    _setup_style()
    output_dir = Path(output_dir)
    if not rule_frames:
        return

    combined = pd.concat(rule_frames, ignore_index=True)
    metric_cols = [c for c in combined.columns if c not in ("mode", "split")]
    melted = combined.melt(id_vars=["mode"], value_vars=metric_cols,
                           var_name="Rule Metric", value_name="Violation (g)")

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_palette = [_PALETTE.get(m, "#999") for m in combined["mode"].unique()]
    sns.barplot(data=melted, x="Rule Metric", y="Violation (g)", hue="mode",
                palette=bar_palette, ax=ax, edgecolor="white")
    ax.set_title("Rule Violation Comparison Across Modes")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.legend(title="Mode", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "rule_violations.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Representation PCA
# ---------------------------------------------------------------------------

def plot_representation_pca(
    model,
    frame: pd.DataFrame,
    feature_columns: list[str],
    image_size: int,
    batch_size: int,
    cache: bool,
    mode: str,
    output_dir: str | Path,
) -> None:
    """PCA of the shared representation colored by Dry_Total_g."""
    _setup_style()
    output_dir = Path(output_dir)

    extractor = build_feature_extractor(model)
    ds = make_prediction_dataset(
        frame, feature_columns,
        image_size=image_size, batch_size=batch_size, cache=cache,
    )
    features = extractor.predict(ds, verbose=0)

    if features.shape[0] < 3:
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(features)
    color_vals = frame["Dry_Total_g"].to_numpy(dtype="float32")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=color_vals, cmap="viridis", s=22, alpha=0.75, edgecolors="white", linewidths=0.3,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Dry Total (g)", fontsize=9)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Shared Representation PCA — {mode.upper()}")
    fig.tight_layout()
    fig.savefig(output_dir / f"representation_pca_{mode}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper called from train.py
# ---------------------------------------------------------------------------

def generate_all_plots(
    run_dir: str | Path,
    history_frames: dict[str, pd.DataFrame],
    metric_frames: list[pd.DataFrame],
    rule_frames: list[pd.DataFrame],
    prediction_data: dict[str, tuple[np.ndarray, np.ndarray]],
    models: dict | None = None,
    test_frame: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
    image_size: int = 128,
    batch_size: int = 16,
    cache: bool = False,
) -> None:
    """Generate all post-training plots and save into run_dir/figures/."""
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Training curves
    if history_frames:
        plot_training_curves(history_frames, fig_dir)

    # Predicted vs Actual
    for mode, (y_true, y_pred) in prediction_data.items():
        plot_pred_vs_actual(y_true, y_pred, mode, fig_dir)

    # Rule violations
    if rule_frames:
        plot_rule_violations(rule_frames, fig_dir)

    # Representation PCA
    if models and test_frame is not None and feature_columns is not None:
        for mode, model in models.items():
            plot_representation_pca(
                model, test_frame, feature_columns,
                image_size=image_size, batch_size=batch_size,
                cache=cache, mode=mode, output_dir=fig_dir,
            )

    print(f"  Saved figures to {fig_dir}")
