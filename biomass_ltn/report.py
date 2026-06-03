from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_run_report(
    output_dir: str | Path,
    config: dict,
    metric_frames: list[pd.DataFrame],
    rule_frames: list[pd.DataFrame],
    representation_rows: list[dict],
    history_frames: dict[str, pd.DataFrame] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    rules = pd.concat(rule_frames, ignore_index=True) if rule_frames else pd.DataFrame()
    representation = pd.DataFrame(representation_rows)

    report_path = output_dir / "run_report.md"
    lines = [
        "# Biomass LTN Run Report",
        "",
        "## Setup",
        "",
        f"- Mode: `{config['mode']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Image size: `{config['image_size']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- LTN loss weight: `{config['ltn_weight']}`",
        f"- LTN warm-up epochs: `{config['ltn_warmup_epochs']}`",
        f"- LTN ramp epochs: `{config['ltn_ramp_epochs']}`",
        f"- Rule tolerance: `{config['rule_tolerance']}`",
        f"- Backbone: `{config['backbone']}`",
        "",
        "## Logical Rules",
        "",
        "- `Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g`",
        "- `GDM_g = Dry_Clover_g + Dry_Green_g`",
        "- all target masses should be non-negative",
        "- derived masses should not be smaller than their components",
        "",
    ]

    # Training history summary
    if history_frames:
        lines.extend(["## Training Summary", ""])
        for mode, df in history_frames.items():
            if df.empty:
                continue
            last = df.iloc[-1]
            lines.append(f"### {mode.upper()}")
            lines.append("")
            lines.append(f"- Epochs trained: {int(last['epoch'])}")
            lines.append(f"- Final train loss: `{last['loss']:.4f}`")
            lines.append(f"- Final train RMSE: `{last['rmse']:.4f}`")
            if "val_loss" in df.columns:
                lines.append(f"- Final val loss: `{last['val_loss']:.4f}`")
                lines.append(f"- Final val RMSE: `{last['val_rmse']:.4f}`")
            if "sat" in df.columns:
                lines.append(f"- Final satisfiability: `{last['sat']:.4f}`")
                lines.append(f"- Final val satisfiability: `{last.get('val_sat', 'N/A')}`")
            best_val_row = df.loc[df["val_rmse"].idxmin()] if "val_rmse" in df.columns else None
            if best_val_row is not None:
                lines.append(f"- Best val RMSE: `{best_val_row['val_rmse']:.4f}` at epoch `{int(best_val_row['epoch'])}`")
            lines.append("")

    if not metrics.empty:
        lines.extend(["## Prediction Metrics", "", metrics.to_markdown(index=False), ""])
    if not rules.empty:
        lines.extend(["## Rule Violations", "", rules.to_markdown(index=False), ""])
    if not representation.empty:
        lines.extend(["## Shared Representation Probe", "", representation.to_markdown(index=False), ""])

    # List generated figures
    fig_dir = output_dir / "figures"
    if fig_dir.exists():
        fig_files = sorted(fig_dir.glob("*.png"))
        if fig_files:
            lines.extend(["## Generated Figures", ""])
            for f in fig_files:
                lines.append(f"- `figures/{f.name}`")
            lines.append("")

    lines.extend(
        [
            "## Reading The Results",
            "",
            "Lower MAE/RMSE means better biomass prediction. Lower rule-violation values mean the predictions better satisfy the pasture biomass identities. The linear probe checks whether the learned shared representation carries target information in a simple, inspectable form.",
            "",
            "## Limitations",
            "",
            "The provided training set is small for image learning, so validation/test estimates can move noticeably across random splits. The symbolic baseline enforces the rules exactly but cannot inspect images. The LTN model trades off supervised fit and consistency, so tune `--ltn-weight` and `--rule-tolerance` for the final report.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
