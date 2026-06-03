#!/usr/bin/env python
"""Exploratory data analysis for the biomass LTN dataset.

Run:
    python analyze_data.py [--data-dir .]

Produces publication-ready figures in analysis_figures/.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET_NAMES = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]

PRETTY_NAMES = {
    "Dry_Clover_g": "Dry Clover",
    "Dry_Dead_g": "Dry Dead",
    "Dry_Green_g": "Dry Green",
    "Dry_Total_g": "Dry Total",
    "GDM_g": "GDM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Biomass dataset EDA")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_figures"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (mirrors biomass_ltn.data but standalone)
# ---------------------------------------------------------------------------

def load_wide_frame(data_dir: Path) -> pd.DataFrame:
    """Load train.csv and pivot from long to wide (one row per image)."""
    csv_path = data_dir / "train.csv"
    images_dir = data_dir / "Images"

    long = pd.read_csv(csv_path)
    long["target"] = pd.to_numeric(long["target"], errors="coerce")

    meta_cols = ["image_path", "Sampling_Date", "State", "Species",
                 "Pre_GSHH_NDVI", "Height_Ave_cm"]
    meta = long[meta_cols].drop_duplicates("image_path").set_index("image_path")
    targets = long.pivot_table(
        index="image_path", columns="target_name", values="target", aggfunc="first",
    )

    frame = meta.join(targets[TARGET_NAMES]).reset_index()
    frame["image_id"] = frame["image_path"].map(lambda v: Path(str(v)).name)
    frame["image_file"] = frame["image_id"].map(lambda n: str(images_dir / n))

    # Parse dates
    frame["Sampling_Date_parsed"] = pd.to_datetime(frame["Sampling_Date"],
                                                    errors="coerce")
    frame["Month"] = frame["Sampling_Date_parsed"].dt.month
    frame["DayOfYear"] = frame["Sampling_Date_parsed"].dt.dayofyear

    # Ensure numeric
    for col in ("Pre_GSHH_NDVI", "Height_Ave_cm"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return frame


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.05, rc={
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    plt.rcParams["font.family"] = "sans-serif"


# ---------------------------------------------------------------------------
# 1. Dataset overview (text)
# ---------------------------------------------------------------------------

def print_overview(df: pd.DataFrame, out: Path):
    lines = [
        "=" * 60,
        "DATASET OVERVIEW",
        "=" * 60,
        f"Total rows (long format CSV): {len(pd.read_csv(out.parent / 'train.csv'))}",
        f"Unique pasture images:        {len(df)}",
        f"Unique States:                {df['State'].nunique()}  {sorted(df['State'].dropna().unique())}",
        f"Unique Species:               {df['Species'].nunique()}",
        "",
        "Target summary statistics:",
        df[TARGET_NAMES].describe().round(2).to_string(),
        "",
        "Missing values per column:",
        df.isnull().sum().to_string(),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    (out / "dataset_overview.txt").write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Target distributions
# ---------------------------------------------------------------------------

def plot_target_distributions(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    colors = sns.color_palette("husl", 5)
    for idx, (name, ax) in enumerate(zip(TARGET_NAMES, axes)):
        values = df[name].dropna()
        ax.hist(values, bins=30, color=colors[idx], edgecolor="white",
                alpha=0.85, density=False)
        sns.kdeplot(values, ax=ax, color="black", linewidth=1.2, warn_singular=False)
        ax.axvline(values.mean(), color="red", linestyle="--", linewidth=1,
                   label=f"Mean={values.mean():.1f}")
        ax.axvline(values.median(), color="blue", linestyle=":", linewidth=1,
                   label=f"Median={values.median():.1f}")
        ax.set_title(PRETTY_NAMES[name], fontsize=11, fontweight="bold")
        ax.set_xlabel("Biomass (g)")
        ax.set_ylabel("Count" if idx == 0 else "")
        ax.legend(fontsize=7, frameon=True)
    fig.suptitle("Target Biomass Distributions", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "target_distributions.png")
    plt.close(fig)
    print("  [OK] target_distributions.png")


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, out: Path):
    corr_cols = TARGET_NAMES + ["Pre_GSHH_NDVI", "Height_Ave_cm"]
    corr = df[corr_cols].corr()
    pretty_labels = [PRETTY_NAMES.get(c, c.replace("_", " ")) for c in corr_cols]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=pretty_labels, yticklabels=pretty_labels,
                ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Feature & Target Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png")
    plt.close(fig)
    print("  [OK] correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Feature distributions
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # NDVI histogram
    ax = axes[0, 0]
    ax.hist(df["Pre_GSHH_NDVI"].dropna(), bins=25, color="#4C72B0",
            edgecolor="white", alpha=0.85)
    ax.set_title("NDVI Distribution", fontweight="bold")
    ax.set_xlabel("Pre GSHH NDVI")

    # Height histogram
    ax = axes[0, 1]
    ax.hist(df["Height_Ave_cm"].dropna(), bins=25, color="#55A868",
            edgecolor="white", alpha=0.85)
    ax.set_title("Average Height Distribution", fontweight="bold")
    ax.set_xlabel("Height (cm)")

    # NDVI by state
    ax = axes[1, 0]
    sns.boxplot(data=df, x="State", y="Pre_GSHH_NDVI", hue="State",
                palette="Set2", width=0.6, fliersize=3, ax=ax, legend=False)
    ax.set_title("NDVI by State", fontweight="bold")

    # Height by state
    ax = axes[1, 1]
    sns.boxplot(data=df, x="State", y="Height_Ave_cm", hue="State",
                palette="Set2", width=0.6, fliersize=3, ax=ax, legend=False)
    ax.set_title("Height by State", fontweight="bold")

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "feature_distributions.png")
    plt.close(fig)
    print("  [OK] feature_distributions.png")


# ---------------------------------------------------------------------------
# 5. Target scatter matrix
# ---------------------------------------------------------------------------

def plot_target_scatter_matrix(df: pd.DataFrame, out: Path):
    subset = df[TARGET_NAMES].dropna()
    pretty = subset.rename(columns=PRETTY_NAMES)
    g = sns.pairplot(pretty, diag_kind="kde", plot_kws={"alpha": 0.45, "s": 12},
                     diag_kws={"linewidth": 1.2},
                     corner=True, height=2.2)
    g.figure.suptitle("Target Pairwise Relationships", fontsize=14,
                      fontweight="bold", y=1.02)
    g.figure.savefig(out / "target_scatter_matrix.png")
    plt.close(g.figure)
    print("  [OK] target_scatter_matrix.png")


# ---------------------------------------------------------------------------
# 6. Additive rule verification in ground truth
# ---------------------------------------------------------------------------

def plot_additive_rules(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rule 1: Dry_Total = Clover + Dead + Green
    ax = axes[0]
    component_sum = df["Dry_Clover_g"] + df["Dry_Dead_g"] + df["Dry_Green_g"]
    ax.scatter(df["Dry_Total_g"], component_sum, alpha=0.5, s=18,
               color="#636EFA", edgecolors="none")
    lim = max(df["Dry_Total_g"].max(), component_sum.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.5)
    residual1 = np.abs(df["Dry_Total_g"] - component_sum)
    ax.set_xlabel("Dry Total (g) — label")
    ax.set_ylabel("Clover + Dead + Green (g)")
    ax.set_title(f"Rule: Total = Sum of Parts\nMAE = {residual1.mean():.3f} g",
                 fontweight="bold", fontsize=10)

    # Rule 2: GDM = Clover + Green
    ax = axes[1]
    gdm_sum = df["Dry_Clover_g"] + df["Dry_Green_g"]
    ax.scatter(df["GDM_g"], gdm_sum, alpha=0.5, s=18,
               color="#00CC96", edgecolors="none")
    lim2 = max(df["GDM_g"].max(), gdm_sum.max()) * 1.1
    ax.plot([0, lim2], [0, lim2], "k--", linewidth=1, alpha=0.5)
    residual2 = np.abs(df["GDM_g"] - gdm_sum)
    ax.set_xlabel("GDM (g) — label")
    ax.set_ylabel("Clover + Green (g)")
    ax.set_title(f"Rule: GDM = Clover + Green\nMAE = {residual2.mean():.3f} g",
                 fontweight="bold", fontsize=10)

    fig.suptitle("Additive Rule Verification in Ground Truth Labels",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "additive_rule_verification.png")
    plt.close(fig)
    print("  [OK] additive_rule_verification.png")


# ---------------------------------------------------------------------------
# 7. PCA of tabular features
# ---------------------------------------------------------------------------

def plot_pca(df: pd.DataFrame, out: Path):
    feature_cols = ["Pre_GSHH_NDVI", "Height_Ave_cm"]
    # Add cyclical date features inline
    dates = pd.to_datetime(df["Sampling_Date"], errors="coerce")
    doy = dates.dt.dayofyear.fillna(dates.dt.dayofyear.median()).astype("float32")
    df_features = df[feature_cols].copy()
    df_features["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df_features["day_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # One-hot encode categoricals
    cats = pd.get_dummies(df[["State", "Species"]].fillna("Unknown"), dtype="float32")
    df_features = pd.concat([df_features, cats], axis=1)

    # Drop rows with NaN
    valid_mask = df_features.notna().all(axis=1)
    df_features = df_features[valid_mask]
    df_valid = df[valid_mask]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_features.astype("float32"))

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Colored by Dry_Total_g
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=df_valid["Dry_Total_g"].values, cmap="viridis",
                    s=20, alpha=0.7, edgecolors="white", linewidths=0.3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Dry Total (g)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA Colored by Dry Total Biomass", fontweight="bold")

    # Colored by Species
    ax = axes[1]
    species = df_valid["Species"].fillna("Unknown")
    unique_species = sorted(species.unique())
    palette = sns.color_palette("husl", len(unique_species))
    for idx, sp in enumerate(unique_species):
        mask = species == sp
        label = sp if len(sp) < 20 else sp[:17] + "..."
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[palette[idx]], s=20, alpha=0.65, label=label,
                   edgecolors="white", linewidths=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA Colored by Species", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6,
              frameon=True, title="Species", title_fontsize=7)

    fig.suptitle("PCA of Tabular Features", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(out / "pca_tabular_features.png")
    plt.close(fig)
    print("  [OK] pca_tabular_features.png")


# ---------------------------------------------------------------------------
# 8. Sample images
# ---------------------------------------------------------------------------

def plot_sample_images(df: pd.DataFrame, out: Path):
    species_groups = df.groupby("Species")
    n_species = min(len(species_groups), 12)
    species_list = sorted(df["Species"].dropna().unique())[:n_species]

    cols = min(4, n_species)
    rows = (n_species + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, sp in enumerate(species_list):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        sample_row = df[df["Species"] == sp].iloc[0]
        img_path = sample_row["image_file"]
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
        title = sp if len(sp) < 25 else sp[:22] + "..."
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # Turn off unused axes
    for idx in range(n_species, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle("Sample Pasture Images by Species", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "sample_images.png")
    plt.close(fig)
    print("  [OK] sample_images.png")


# ---------------------------------------------------------------------------
# 9. Biomass by Species & State
# ---------------------------------------------------------------------------

def plot_biomass_by_category(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    state_means = df.groupby("State")[TARGET_NAMES].mean()
    state_means.rename(columns=PRETTY_NAMES).plot(
        kind="bar", ax=ax, width=0.8, edgecolor="white")
    ax.set_title("Mean Biomass by State", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Biomass (g)")
    ax.legend(fontsize=7, frameon=True)
    ax.tick_params(axis="x", rotation=0)

    ax = axes[1]
    species_total = df.groupby("Species")["Dry_Total_g"].agg(["mean", "std", "count"])
    species_total = species_total.sort_values("mean", ascending=True)
    species_total = species_total.tail(15)  # top 15
    labels = [s if len(s) < 20 else s[:17] + "..." for s in species_total.index]
    ax.barh(range(len(species_total)), species_total["mean"],
            xerr=species_total["std"], color="#4C72B0", edgecolor="white",
            alpha=0.85, capsize=3)
    ax.set_yticks(range(len(species_total)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Dry Total (g)")
    ax.set_title("Mean Dry Total Biomass by Species (Top 15)", fontweight="bold")

    fig.suptitle("Biomass Distribution by Category", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "biomass_by_category.png")
    plt.close(fig)
    print("  [OK] biomass_by_category.png")


# ---------------------------------------------------------------------------
# 10. Summary statistics CSV
# ---------------------------------------------------------------------------

def save_summary_stats(df: pd.DataFrame, out: Path):
    stats = df[TARGET_NAMES].describe().T
    stats["median"] = df[TARGET_NAMES].median()
    stats["skew"] = df[TARGET_NAMES].skew()
    stats = stats[["count", "mean", "std", "min", "25%", "median", "75%", "max", "skew"]]
    stats.index = [PRETTY_NAMES.get(n, n) for n in stats.index]
    stats.to_csv(out / "summary_statistics.csv")
    print("  [OK] summary_statistics.csv")
    print()
    print(stats.round(2).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_style()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir} ...")
    df = load_wide_frame(args.data_dir)
    print(f"Loaded {len(df)} images.\n")

    print_overview(df, args.data_dir)
    print("\nGenerating figures ...")
    plot_target_distributions(df, out)
    plot_correlation_heatmap(df, out)
    plot_feature_distributions(df, out)
    plot_target_scatter_matrix(df, out)
    plot_additive_rules(df, out)
    plot_pca(df, out)
    plot_sample_images(df, out)
    plot_biomass_by_category(df, out)

    print("\nSummary Statistics:")
    save_summary_stats(df, out)

    print(f"\nAll outputs saved to {out.resolve()}")


if __name__ == "__main__":
    main()
