#!/usr/bin/env python
"""Generate a professional technical report PDF for the Biomass LTN project.

Uses fpdf2 to produce a self-contained PDF with embedded figures from the
best training run.

Usage:
    python generate_report_pdf.py
"""
from __future__ import annotations

import os
from pathlib import Path

from fpdf import FPDF

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs" / "20260603-180836"
FIG_DIR = RUN_DIR / "figures"
OUTPUT_PDF = ROOT / "Technical_Report.pdf"

# ── Colour palette ─────────────────────────────────────────────────────────
C_TITLE = (30, 58, 95)        # dark navy
C_HEADING = (40, 75, 120)     # steel blue
C_SUBHEAD = (60, 100, 145)    # medium blue
C_BODY = (30, 30, 30)         # near-black
C_ACCENT = (0, 120, 180)      # teal accent
C_LIGHT_BG = (240, 245, 250)  # light grey-blue for table rows


class ReportPDF(FPDF):
    """Custom PDF with headers, footers, and helper methods."""

    def header(self):
        if self.page_no() == 1:
            return  # title page has its own header
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(
            0, 6,
            "Neuro-Symbolic Biomass Prediction with Logical Tensor Networks",
            align="L",
        )
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── helpers ─────────────────────────────────────────────────────────
    def section_title(self, number: int, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*C_HEADING)
        self.cell(0, 9, f"{number}.  {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.l_margin + 55, self.get_y())
        self.ln(4)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*C_SUBHEAD)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet(self, text: str, indent: float = 8):
        x0 = self.get_x()
        self.set_x(x0 + indent)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        self.cell(4, 5, "-")
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_inline(self, label: str, text: str):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*C_BODY)
        self.cell(self.get_string_width(label) + 1, 5, label)
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def add_figure(self, path: str, caption: str, w: float = 170):
        if not os.path.exists(path):
            self.body(f"[Figure not found: {path}]")
            return
        available = self.h - self.get_y() - self.b_margin - 12
        if available < 50:
            self.add_page()
        x = (self.w - w) / 2
        self.image(path, x=x, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 4, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def add_table(self, headers: list[str], rows: list[list[str]],
                  col_widths: list[float] | None = None):
        if col_widths is None:
            cw = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [cw] * len(headers)
        # header row
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*C_HEADING)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align="C", fill=True)
        self.ln()
        # data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*C_BODY)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(*C_LIGHT_BG)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, val, border=1, align="C", fill=True)
            self.ln()
        self.ln(3)


def build_pdf():
    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ═══════════════════════════════════════════════════════════════════
    # TITLE HEADER (compact — shares page 1 with Introduction)
    # ═══════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*C_TITLE)
    pdf.cell(0, 9, "Neuro-Symbolic Biomass Prediction with", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 9, "Logical Tensor Networks", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(*C_ACCENT)
    pdf.set_line_width(0.8)
    mid = pdf.w / 2
    pdf.line(mid - 35, pdf.get_y(), mid + 35, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Himal Garg  |  June 2026  |  Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ═══════════════════════════════════════════════════════════════════
    # 1. INTRODUCTION & OBJECTIVE
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(1, "Introduction & Objective")
    pdf.body(
        "Accurate estimation of pasture biomass is essential for livestock "
        "management and feed budgeting in precision agriculture. Traditional "
        "methods require destructive sampling, which is time-consuming and "
        "impractical at scale. This project explores an automated, non-destructive "
        "approach that predicts five biomass components from pasture images and "
        "associated metadata."
    )
    pdf.body(
        "The five target variables are: Dry Clover (g), Dry Dead (g), "
        "Dry Green (g), Dry Total (g), and Green Dry Matter - GDM (g). These "
        "targets are not independent: they obey known agronomic identities "
        "(e.g., Dry Total = Clover + Dead + Green). A standard neural "
        "network can learn to predict each target but may produce physically "
        "inconsistent outputs. A purely symbolic model can enforce the equations "
        "but cannot learn from raw image data."
    )
    pdf.body(
        "Logical Tensor Networks (LTNs) bridge this gap. By grounding a neural "
        "regression model as a differentiable function within a first-order "
        "fuzzy logic framework, LTNs allow us to inject domain knowledge "
        "as soft logical axioms directly into the training loss. The model "
        "thus learns to be both perceptually accurate and logically consistent."
    )

    # ═══════════════════════════════════════════════════════════════════
    # 2. EXPERIMENTAL SETUP
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(2, "Experimental Setup")

    pdf.sub_title("2.1  Dataset")
    pdf.body(
        "The dataset consists of 357 unique pasture plot images, each associated "
        "with tabular metadata (NDVI, average plant height, geographical state, "
        "and species). The original train.csv contains five long-format rows per "
        "image; these are pivoted into a single five-output regression sample per "
        "image. Total samples after pivoting: 357."
    )

    pdf.sub_title("2.2  State-Stratified Data Splitting")
    pdf.body(
        "Pasture records exhibit significant geographical class imbalance across "
        "Australian states. To ensure robust evaluation, the data is split into "
        "Train (70%, n=249), Validation (15%, n=54), and Test (15%, n=54) sets "
        "using stratification on the State column. A fallback mechanism handles "
        "states with fewer than 3 samples to prevent splitting errors."
    )

    pdf.sub_title("2.3  Feature & Target Normalization")
    pdf.body(
        "Tabular features are z-score normalized using training-set statistics. "
        "Biomass targets are divided by a single shared global scaling factor "
        "(166.1 g, the max training target) to preserve the additive linear "
        "structure in normalized space. This is critical: if Total = Clover + "
        "Dead + Green holds in grams, it also holds after division by a common "
        "constant."
    )

    pdf.sub_title("2.4  Model Architecture")
    pdf.body(
        "The model is a two-stream multimodal fusion network: (1) a Visual Stream "
        "using MobileNetV2 (pre-trained on ImageNet) processes 160x160 pasture "
        "images, and (2) a Tabular Stream of dense layers encodes 24-dimensional "
        "metadata. The two embeddings are concatenated into a shared representation "
        "and fed through a 5-output regression head with softplus activation."
    )
    pdf.body(
        "Softplus (f(x) = log(1 + exp(x))) replaces sigmoid and ReLU to avoid "
        "output saturation and dead neuron problems. The output bias is initialized "
        "with the inverse softplus of training target means (b = log(exp(mu) - 1)) "
        "for stable early training."
    )

    pdf.sub_title("2.5  Loss Function")
    pdf.body(
        "Huber Loss (delta = 0.15) is used instead of MSE to handle the skewed "
        "target distributions and extreme outliers present in biomass data. Huber "
        "Loss behaves as L2 near zero and L1 for large errors, combining the "
        "stability of squared loss with the robustness of absolute loss."
    )

    pdf.sub_title("2.6  Training Configuration")
    pdf.add_table(
        ["Parameter", "Value"],
        [
            ["Backbone", "MobileNetV2 (ImageNet)"],
            ["Image Size", "160 x 160"],
            ["Batch Size", "8"],
            ["Epochs", "50 (early stopping, patience=10)"],
            ["Learning Rate", "0.001"],
            ["Dropout", "0.2"],
            ["LTN Weight", "0.01"],
            ["Rule Tolerance", "0.12"],
            ["LTN Warmup Epochs", "5"],
            ["LTN Ramp Epochs", "5"],
            ["Base Loss", "Huber (delta=0.15)"],
            ["Output Activation", "Softplus"],
            ["Seed", "42"],
        ],
        col_widths=[60, 120],
    )

    # ═══════════════════════════════════════════════════════════════════
    # 3. COMPARATIVE ANALYSIS — THREE MODES
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(3, "Comparative Analysis")
    pdf.body(
        "Three learning modes are compared on identical train/val/test splits:"
    )

    pdf.sub_title("3.1  Symbolic Baseline")
    pdf.body(
        "Predicts primitive biomass components (Clover, Dead, Green) using "
        "group-mean estimates from metadata categories, then derives Total and "
        "GDM exactly from the additive equations. This baseline has zero rule "
        "violations by construction but limited predictive power because it "
        "ignores image evidence entirely."
    )

    pdf.sub_title("3.2  Neural Model (Huber Loss Only)")
    pdf.body(
        "Trains the full multimodal fusion network using only Huber regression "
        "loss. No logical constraints are applied. The model freely optimizes "
        "each of the five targets independently, which can produce physically "
        "inconsistent predictions (e.g., Total < Green)."
    )

    pdf.sub_title("3.3  LTN Model (Neuro-Symbolic)")
    pdf.body(
        "Trains the same architecture with a combined loss: Huber regression "
        "plus a weighted satisfiability penalty from the LTN axiom knowledge "
        "base. The LTN loss is warmed up after 5 supervised-only epochs and "
        "linearly ramped over the next 5 epochs to prevent shortcut learning "
        "(collapsing all predictions to zero to trivially satisfy axioms)."
    )

    # ═══════════════════════════════════════════════════════════════════
    # 4. INTERPRETATION OF FUZZY LOGICS
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(4, "Interpretation of Fuzzy Logics")
    pdf.body(
        "The LTN framework translates classical first-order logic axioms into "
        "differentiable fuzzy operations over real-valued tensors. Each axiom "
        "produces a truth value in [0, 1], where 1 means fully satisfied and "
        "0 means maximally violated."
    )

    pdf.sub_title("4.1  Regression Predicates")
    pdf.body(
        "A continuous regression predicate converts numerical agreement into a "
        "truth degree. For data fitting, truth is high when the predicted and "
        "observed values are close (relative to target-specific tolerances "
        "estimated from the training split). For rule matching, truth decays "
        "smoothly as the equation residual grows beyond the rule tolerance "
        "(0.12 in normalized space)."
    )

    pdf.sub_title("4.2  The Knowledge Base Axioms")
    pdf.body("The following axioms form the LTN knowledge base:")
    pdf.bullet("Data Fitting: For all x, predicted values Close(f(x), y) should match observed targets.")
    pdf.bullet("Total Mass Decomposition: Dry Total (g) = Dry Clover (g) + Dry Dead (g) + Dry Green (g).")
    pdf.bullet("GDM Composition: GDM (g) = Dry Clover (g) + Dry Green (g).")
    pdf.bullet("Non-negativity: All predicted masses must be >= 0.")
    pdf.bullet("Part-Whole Ordering: Aggregate masses (Total, GDM) must be >= each component mass.")

    pdf.sub_title("4.3  Satisfiability Aggregation")
    pdf.body(
        "Individual axiom truth values are aggregated using the p-mean error "
        "operator (from the LTN reference implementation). The batch "
        "satisfiability score Sat(Phi) represents the overall logical "
        "consistency of predictions. The combined loss is: "
        "L_total = L_Huber + lambda_ltn * (1 - Sat(Phi)), where lambda_ltn = 0.01."
    )
    pdf.body(
        "The soft nature of fuzzy logic is crucial for noisy biological "
        "measurements. A prediction with a small residual is not rejected; "
        "it receives a truth value slightly below 1. Only large violations "
        "produce low truth values and significant gradient pressure. This "
        "provides a smooth, differentiable signal that guides the model "
        "toward consistency without brittle hard constraints."
    )

    # ═══════════════════════════════════════════════════════════════════
    # 5. RESULTS
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(5, "Results")

    pdf.sub_title("5.1  Regression Performance (Test Set)")
    pdf.add_table(
        ["Metric", "Symbolic", "Neural", "LTN"],
        [
            ["MAE (g) - Overall",     "8.75",   "10.69",  "9.31"],
            ["RMSE (g) - Overall",    "15.15",  "15.59",  "15.37"],
            ["R2 - Overall",          "0.470",  "0.440",  "0.347"],
            ["R2 - Dry Green",        "0.636",  "0.455",  "0.632"],
            ["R2 - GDM",              "0.506",  "0.570",  "0.668"],
            ["R2 - Dry Total",        "0.511",  "0.551",  "0.451"],
            ["R2 - Dry Clover",       "0.336",  "0.282",  "0.024"],
            ["R2 - Dry Dead",         "0.363",  "0.340",  "-0.040"],
        ],
        col_widths=[50, 35, 35, 35],
    )

    pdf.sub_title("5.2  Logical Consistency (Rule Violations, Test Set)")
    pdf.add_table(
        ["Violation Metric", "Symbolic", "Neural", "LTN", "Reduction"],
        [
            ["Total Sum MAE (g)",    "0.00",  "15.12",  "3.98",   "74%"],
            ["GDM Sum MAE (g)",      "0.00",  "11.26",  "4.84",   "57%"],
            ["Negative Violation (g)","0.00",  "0.00",   "0.00",   "-"],
            ["Order Violation (g)",  "0.00",  "2.64",   "0.19",   "93%"],
        ],
        col_widths=[40, 28, 28, 28, 28],
    )

    pdf.body(
        "The LTN model achieves the best GDM R-squared (0.668) and matches "
        "the symbolic baseline on Dry Green R-squared (0.632 vs 0.636), while "
        "dramatically reducing rule violations compared to the pure neural "
        "model. Total sum violations drop by 74%, GDM violations by 57%, "
        "and ordering violations by 93%."
    )

    # Figures
    pdf.sub_title("5.3  Training Dynamics")
    pdf.add_figure(
        str(FIG_DIR / "loss_curves.png"),
        "Fig 1: Training and validation loss curves for Neural and LTN models.",
        w=115,
    )
    pdf.add_figure(
        str(FIG_DIR / "satisfiability_curve.png"),
        "Fig 2: LTN satisfiability score over training epochs.",
        w=105,
    )

    pdf.sub_title("5.4  Predicted vs. Actual (Test Set)")
    pdf.add_figure(
        str(FIG_DIR / "pred_vs_actual_ltn.png"),
        "Fig 3: LTN model - predicted vs. actual biomass (test set).",
        w=125,
    )
    pdf.add_figure(
        str(FIG_DIR / "pred_vs_actual_neural.png"),
        "Fig 4: Neural model - predicted vs. actual biomass (test set).",
        w=125,
    )

    pdf.sub_title("5.5  Rule Violations Comparison")
    pdf.add_figure(
        str(FIG_DIR / "rule_violations.png"),
        "Fig 5: Rule violation comparison across all three modes.",
        w=115,
    )
    pdf.body(
        "Linear probe analysis: Neural probe R2 = 0.640, LTN probe R2 "
        "= 0.615. The LTN representation trades slight linear predictability "
        "for dramatically better constraint satisfaction."
    )

    # ═══════════════════════════════════════════════════════════════════
    # 6. LIMITATIONS
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(6, "Limitations")
    pdf.bullet(
        "Small dataset: With only 357 images, test metrics can vary "
        "noticeably depending on the random seed and split. Cross-validation "
        "would provide more stable estimates but was not implemented."
    )
    pdf.bullet(
        "Clover and Dead targets: The LTN model achieves near-zero or "
        "negative R-squared for Dry Clover and Dry Dead. Clover is "
        "approximately 38% zero-valued in the dataset, making it inherently "
        "difficult to predict. The logical constraints may be diverting "
        "capacity away from these sparse targets toward better consistency "
        "on the aggregate targets."
    )
    pdf.bullet(
        "Known-identity axioms: The logical rules encode known target "
        "identities, so the LTN layer improves consistency rather than "
        "discovering new biological laws. If labels contain rounding "
        "errors, strict tolerances can hurt supervised accuracy."
    )
    pdf.bullet(
        "Transfer learning sensitivity: MobileNetV2 ImageNet features "
        "were pre-trained on natural images, not agricultural plots. "
        "Domain-specific pre-training could improve visual feature quality."
    )
    pdf.bullet(
        "Shortcut learning risk: Without careful tuning of lambda_ltn and "
        "the warmup schedule, the model can collapse to predicting zero "
        "for all components (trivially satisfying sum constraints)."
    )

    # ═══════════════════════════════════════════════════════════════════
    # 7. FUTURE WORK
    # ═══════════════════════════════════════════════════════════════════
    pdf.section_title(7, "Directions for Future Work")
    pdf.bullet(
        "Cross-validation: Implement k-fold cross-validation stratified by "
        "State to reduce variance in performance estimates."
    )
    pdf.bullet(
        "Target transformations: Apply log or Box-Cox transforms to zero-"
        "inflated targets (Clover) to improve predictions for sparse "
        "components. Rules must then be evaluated in linear space by "
        "inverting the transform before computing axiom truth values."
    )
    pdf.bullet(
        "Learnable rule weights: Instead of a fixed lambda_ltn, learn "
        "per-axiom weights to let the model express confidence in each "
        "rule. This could prevent well-satisfied axioms from dominating "
        "the loss signal."
    )
    pdf.bullet(
        "Uncertainty estimation: Add Monte Carlo dropout or ensemble "
        "methods to quantify prediction uncertainty, especially for "
        "rare pasture types."
    )
    pdf.bullet(
        "Domain-specific pre-training: Fine-tune the visual backbone on "
        "agricultural imagery datasets before training on this dataset."
    )
    pdf.bullet(
        "Calibration curves: Analyze how fuzzy predicate truth values "
        "correlate with actual prediction error to validate the semantic "
        "meaning of the satisfiability score."
    )
    pdf.bullet(
        "Ablation studies: Systematically disable individual axioms to "
        "measure the marginal contribution of each rule to overall "
        "performance and consistency."
    )

    # ═══════════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════════
    pdf.output(str(OUTPUT_PDF))
    print(f"\nPDF report saved to: {OUTPUT_PDF}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_pdf()
