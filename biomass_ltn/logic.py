from __future__ import annotations

import numpy as np
import tensorflow as tf

import ltn
from ltn.core import Formula

from .config import TARGET_INDEX


class BiomassLTNObjective:
    """LTN grounding for supervised regression plus biomass consistency rules."""

    def __init__(
        self,
        model: tf.keras.Model,
        target_tolerances: np.ndarray,
        rule_tolerance: float = 0.03,
        aggregation_p: int = 2,
    ) -> None:
        self.model = model
        self.regressor = ltn.Function(model)
        self.target_tolerances = tf.constant(target_tolerances, dtype=tf.float32)
        self.rule_tolerance = tf.constant(rule_tolerance, dtype=tf.float32)

        self.Forall = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=aggregation_p),
            semantics="forall",
        )
        self.sample_aggregator = ltn.fuzzy_ops.Aggreg_pMeanError(p=aggregation_p)

        self.matches_targets = ltn.Predicate.Lambda(self._matches_targets)
        self.total_is_sum = ltn.Predicate.Lambda(self._total_is_sum)
        self.gdm_is_green_plus_clover = ltn.Predicate.Lambda(self._gdm_is_green_plus_clover)
        self.non_negative = ltn.Predicate.Lambda(self._non_negative)
        self.order_consistent = ltn.Predicate.Lambda(self._order_consistent)

    def axioms(
        self,
        images: tf.Tensor,
        tabular: tf.Tensor,
        y_true: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        """Return the batch-level satisfiability of the grounded knowledge base."""
        image_var = ltn.Variable("image", images)
        tabular_var = ltn.Variable("tabular", tabular)
        target_var = ltn.Variable("target", y_true)
        image_var, tabular_var, target_var = ltn.diag(image_var, tabular_var, target_var)

        y_pred = self.regressor([image_var, tabular_var], training=training)
        formulas = [
            self.matches_targets([y_pred, target_var]),
            self.total_is_sum([y_pred]),
            self.gdm_is_green_plus_clover([y_pred]),
            self.non_negative([y_pred]),
            self.order_consistent([y_pred]),
        ]
        stacked_truth = tf.stack([formula.tensor for formula in formulas], axis=0)
        sample_truth = Formula(
            self.sample_aggregator(stacked_truth, axis=0),
            formulas[0].free_vars,
        )
        return self.Forall([image_var, tabular_var, target_var], sample_truth).tensor

    def axioms_from_predictions(
        self,
        y_pred: tf.Tensor,
        y_true: tf.Tensor,
    ) -> tf.Tensor:
        """Compute satisfiability from pre-computed predictions (avoids double forward pass)."""
        pred_var = ltn.Variable("pred", y_pred)
        target_var = ltn.Variable("target", y_true)
        pred_var, target_var = ltn.diag(pred_var, target_var)

        formulas = [
            self.matches_targets([pred_var, target_var]),
            self.total_is_sum([pred_var]),
            self.gdm_is_green_plus_clover([pred_var]),
            self.non_negative([pred_var]),
            self.order_consistent([pred_var]),
        ]
        stacked_truth = tf.stack([formula.tensor for formula in formulas], axis=0)
        sample_truth = Formula(
            self.sample_aggregator(stacked_truth, axis=0),
            formulas[0].free_vars,
        )
        return self.Forall([pred_var, target_var], sample_truth).tensor

    def _matches_targets(self, args: list[tf.Tensor]) -> tf.Tensor:
        y_pred, y_true = args
        scaled_abs_error = tf.abs(y_pred - y_true) / (self.target_tolerances + 1e-6)
        return 1.0 / (1.0 + tf.reduce_mean(scaled_abs_error, axis=-1))

    def _total_is_sum(self, args: list[tf.Tensor]) -> tf.Tensor:
        pred = args[0]
        clover = pred[..., TARGET_INDEX["Dry_Clover_g"]]
        dead = pred[..., TARGET_INDEX["Dry_Dead_g"]]
        green = pred[..., TARGET_INDEX["Dry_Green_g"]]
        total = pred[..., TARGET_INDEX["Dry_Total_g"]]
        return self._close_truth(total, clover + dead + green)

    def _gdm_is_green_plus_clover(self, args: list[tf.Tensor]) -> tf.Tensor:
        pred = args[0]
        clover = pred[..., TARGET_INDEX["Dry_Clover_g"]]
        green = pred[..., TARGET_INDEX["Dry_Green_g"]]
        gdm = pred[..., TARGET_INDEX["GDM_g"]]
        return self._close_truth(gdm, clover + green)

    def _non_negative(self, args: list[tf.Tensor]) -> tf.Tensor:
        pred = args[0]
        violation = tf.nn.relu(-pred)
        truth = tf.exp(-violation / (self.rule_tolerance + 1e-6))
        return tf.reduce_min(truth, axis=-1)

    def _order_consistent(self, args: list[tf.Tensor]) -> tf.Tensor:
        pred = args[0]
        clover = pred[..., TARGET_INDEX["Dry_Clover_g"]]
        dead = pred[..., TARGET_INDEX["Dry_Dead_g"]]
        green = pred[..., TARGET_INDEX["Dry_Green_g"]]
        total = pred[..., TARGET_INDEX["Dry_Total_g"]]
        gdm = pred[..., TARGET_INDEX["GDM_g"]]

        violations = tf.stack(
            [
                tf.nn.relu(clover - total),
                tf.nn.relu(dead - total),
                tf.nn.relu(green - total),
                tf.nn.relu(gdm - total),
                tf.nn.relu(clover - gdm),
                tf.nn.relu(green - gdm),
            ],
            axis=-1,
        )
        return tf.exp(-tf.reduce_mean(violations, axis=-1) / (self.rule_tolerance + 1e-6))

    def _close_truth(self, lhs: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
        return tf.exp(-tf.abs(lhs - rhs) / (self.rule_tolerance + 1e-6))


def target_tolerances_from_frame(frame, target_names, target_scale: float) -> np.ndarray:
    values = frame[target_names].to_numpy(dtype="float32") / np.float32(target_scale)
    tolerances = np.nanstd(values, axis=0)
    return np.maximum(tolerances, 0.03).astype("float32")

