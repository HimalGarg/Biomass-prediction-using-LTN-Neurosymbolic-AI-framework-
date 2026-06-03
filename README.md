# Biomass Logical Tensor Network

This project implements the PDF brief for neuro-symbolic biomass prediction with
Logical Tensor Networks. It uses the local `train.csv` and `Images/` folder, plus
the cloned reference implementation at `external/logictensornetworks`.

## What Is Built

- image + tabular metadata fusion model
- five-output biomass regression head
- LTN objective using the official TensorFlow LTN package
- soft biomass consistency rules:
  - `Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g`
  - `GDM_g = Dry_Clover_g + Dry_Green_g`
  - non-negative masses
  - derived masses are at least as large as their components
- neural-only and symbolic-only baselines
- per-target metrics, rule-violation metrics, prediction CSVs, and generated run reports
- auto-generated training curves, predicted-vs-actual scatters, and rule-violation bar charts
- data analysis script with EDA, PCA, and correlation analysis

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

The current workspace already has the virtual environment prepared.

## Data Analysis

Run the standalone EDA script to produce analysis figures:

```powershell
.\.venv\Scripts\python analyze_data.py
```

This generates publication-ready figures in `analysis_figures/`:
- target distributions with KDE
- correlation heatmap (targets + features)
- feature distributions and boxplots by State
- target pairwise scatter matrix
- additive rule verification on ground-truth labels
- PCA of tabular features colored by biomass and species
- sample images by species
- biomass breakdown by category

## Quick Smoke Test

```powershell
.\.venv\Scripts\python train.py --mode ltn --epochs 1 --image-size 64 --batch-size 8 --limit-samples 40 --no-augment
```

## Full Comparison

```powershell
.\.venv\Scripts\python train.py --mode all --epochs 40 --image-size 128 --batch-size 16
```

The LTN run uses a supervised warm-up by default, then ramps in the rule loss. To make the
logic stricter, increase `--ltn-weight`; to make the equations softer, increase
`--rule-tolerance`. Early stopping is enabled by default (`--patience 10`); set
`--patience 0` to disable.

For a stronger visual encoder, use MobileNetV2:

```powershell
.\.venv\Scripts\python train.py --mode all --epochs 40 --image-size 160 --batch-size 8 --backbone mobilenetv2 --mobilenet-weights imagenet
```

## Outputs

Each run creates a timestamped folder under `runs/` containing:

- `config.json`
- `metrics_all.csv`
- `rules_all.csv`
- `representation_probe.csv`
- `training_history_neural.csv` and/or `training_history_ltn.csv`
- `predictions/*.csv`
- `models/neural_weights.h5` and/or `models/ltn_weights.h5`
- `figures/` — training curves, pred-vs-actual scatters, rule violations, representation PCA
- `run_report.md`

The longer design write-up is in `REPORT.md`.
