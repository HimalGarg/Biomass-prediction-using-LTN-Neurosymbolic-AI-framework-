# Neuro-Symbolic Biomass Prediction With Logical Tensor Networks

## 1. Objective

This project implements the assignment brief: predict pasture biomass from images and
metadata using a neuro-symbolic model based on Logical Tensor Networks (LTNs). The target
variables are `Dry_Clover_g`, `Dry_Dead_g`, `Dry_Green_g`, `Dry_Total_g`, and `GDM_g`.
The available data contains 357 unique pasture images, with five long-format rows in
`train.csv` for each image. The code pivots those rows into a single five-output regression
example per image.

The central design goal is not only to minimize prediction error, but also to produce
predictions that respect known biomass identities. A purely neural model can predict each
target independently and violate basic equations. A purely symbolic model can enforce the
equations but cannot inspect the image. The LTN approach combines both: perception comes
from a neural network, and consistency is encouraged through differentiable logical truth
values.

## 2. Data Processing

The local CSV contains image references like `train/ID1011485656.jpg`; these are resolved
against the local `Images/` directory. Metadata is converted into model-ready features:

- numeric measurements: `Pre_GSHH_NDVI`, `Height_Ave_cm`
- date encodings: sine/cosine day-of-year and sampling year
- categorical metadata: one-hot `State` and `Species`

The model uses a deterministic train/validation/test split. Tabular features are standardized
using only the training split. Targets are divided by one shared training biomass scale. A
single shared target scale is important because it preserves additive equations after
normalization. For example, if every target is divided by the same value, then
`Dry_Total = Clover + Dead + Green` remains true in normalized space.

## 3. Multimodal Neural Model

The primary neural architecture is a two-stream model:

- an image encoder processes resized pasture images
- a tabular encoder processes metadata
- both embeddings are concatenated into a shared fused representation
- a five-output sigmoid regression head predicts normalized biomass values

The default image encoder is a compact CNN so the project runs on a CPU without downloading
external weights. The model also supports MobileNetV2 with ImageNet weights for stronger
visual experiments:

```powershell
.\.venv\Scripts\python train.py --mode all --epochs 30 --image-size 160 --batch-size 8 --backbone mobilenetv2 --mobilenet-weights imagenet
```

The final layer is sigmoid-bounded in normalized biomass space and its bias is initialized
from the normalized training-set target means. This avoids the degenerate all-zero saturation
that can happen when a positive-only output head is initialized far above sparse biomass
targets. Non-negativity is still reported explicitly as a rule-violation metric.

## 4. LTN Grounding

The implementation uses the TensorFlow LTN code from
`external/logictensornetworks`, cloned from `logictensornetworks/logictensornetworks`.
The Keras regressor is grounded as an LTN function:

```text
f(image, metadata) -> [clover, dead, green, total, gdm]
```

Continuous regression predicates convert numerical agreement into truth degrees in `[0, 1]`.
For target matching, the predicate is high when predicted and observed biomass are close
relative to target-specific tolerances estimated from the training split. For rule matching,
truth decays smoothly as the equation residual grows.

The knowledge base contains these soft axioms:

```text
forall x: observed_targets_match(f(x), y)
forall x: Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g
forall x: GDM_g = Dry_Clover_g + Dry_Green_g
forall x: all predicted masses are non-negative
forall x: derived masses are not smaller than their components
```

The batch satisfiability score is aggregated with the p-mean-error operator used in the LTN
reference examples. Training minimizes:

```text
loss = supervised_mse + ltn_weight * (1 - satisfiability)
```

This objective gives the model two pressures: fit the labels and keep the predictions
semantically coherent. The logical penalty is warmed up gradually: the first few epochs are
supervised-only, then the rule loss ramps in. This prevents the model from choosing a
trivial low-biomass solution before it has learned the target scale.

## 5. Fuzzy Logic Interpretation

The LTN rules are soft rather than hard. A prediction with a small total-mass residual is not
discarded; it receives a truth value slightly below 1. Larger violations receive lower truth
values and therefore increase the loss. This is useful for noisy biological measurements,
where exact equality may be too brittle.

The two additive equations have direct semantic meaning:

- dry total biomass should equal clover, dead, and green dry biomass
- green dry matter should equal green biomass plus clover biomass

The inequality rules are secondary checks. They help prevent impossible outputs such as a
component mass larger than its derived aggregate. The non-negativity rule makes the intended
domain assumption explicit and measurable.

## 6. Baseline Comparisons

The code includes three modes:

- `symbolic`: predicts primitive masses from metadata group means, then derives total and
  GDM exactly from the symbolic equations
- `neural`: trains the multimodal neural network with MSE only
- `ltn`: trains the same multimodal network with MSE plus LTN satisfiability loss

The symbolic baseline is expected to have zero rule violation but limited predictive power,
because it does not use image evidence. The neural baseline may learn useful image and
metadata patterns, but it can produce inconsistent target combinations. The LTN model should
reduce rule violations while preserving as much predictive accuracy as possible.

## 7. Evaluation Protocol

Each run writes a timestamped directory under `runs/` containing model artifacts, predictions,
metrics, and a generated `run_report.md`. The evaluation tracks:

- overall and per-target MAE
- overall and per-target RMSE
- overall and per-target R2
- `Dry_Total_g` equation violation in grams
- `GDM_g` equation violation in grams
- non-negativity violation
- aggregate/component order violation
- linear-probe quality of the learned `shared_representation`

The linear probe freezes the learned shared representation, trains a Ridge regressor on top,
and measures whether the representation is linearly predictive of biomass. This does not
replace task metrics, but it gives a compact diagnostic for representation quality.

## 8. Reproduction Commands

Quick LTN smoke test:

```powershell
.\.venv\Scripts\python train.py --mode ltn --epochs 1 --image-size 64 --batch-size 8 --limit-samples 40 --no-augment
```

Full compact-CNN comparison:

```powershell
.\.venv\Scripts\python train.py --mode all --epochs 30 --image-size 128 --batch-size 16
```

The compact CNN is the recommended first full run because it is reproducible and does not
need network access after dependencies are installed. MobileNetV2 is recommended for the
stronger final experiment if runtime allows.

## 9. Limitations

The dataset is small for end-to-end visual learning. With only 357 images, the test metrics
can vary noticeably depending on split and seed. The compact CNN may underfit visual details,
while MobileNetV2 can improve features but adds dependence on pretrained image statistics.

The logical equations are known target identities, so the LTN layer improves consistency
rather than discovering new biological laws. If the labels themselves contain noise or
rounding differences, overly strict rule tolerances can hurt supervised accuracy. The best
`ltn_weight` and `rule_tolerance` should therefore be tuned and reported, not assumed.

The symbolic baseline is intentionally simple. It is useful as a consistency reference, but
it should not be interpreted as a strong agronomic model.

## 10. Future Work

Useful extensions include cross-validation, pretrained visual backbones, uncertainty
estimation, calibration curves for the fuzzy predicates, and ablations over each logical
rule. Another useful direction is to learn rule weights so the model can express confidence
in each axiom while still respecting known biomass structure.