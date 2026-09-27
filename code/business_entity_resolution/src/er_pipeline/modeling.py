"""Ownership classifier and precision-heavy threshold selection."""

from __future__ import annotations

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier

from .features import FEATURE_COLUMNS


def make_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=750, learning_rate=0.035, num_leaves=63,
        min_child_samples=60, max_bin=255, colsample_bytree=0.9,
        subsample=0.9, subsample_freq=1, reg_alpha=0.25, reg_lambda=3.0,
        random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _top_with_margin(scored: pl.DataFrame) -> pl.DataFrame:
    ranked = scored.sort(
        ["target_key", "probability"], descending=[False, True]
    ).with_columns(
        pl.col("probability").shift(-1).over("target_key").fill_null(0.0)
        .alias("second_probability")
    )
    return ranked.group_by("target_key", maintain_order=True).head(1).with_columns(
        (pl.col("probability") - pl.col("second_probability"))
        .alias("probability_margin")
    )


def target_decisions(scored: pl.DataFrame, threshold: float,
                     minimum_margin: float = 0.0) -> pl.DataFrame:
    return _top_with_margin(scored).filter(
        (pl.col("probability") >= threshold)
        & (pl.col("probability_margin") >= minimum_margin)
    )


def tune_threshold(scored: pl.DataFrame, target_truth: pl.DataFrame) -> tuple[float, float, dict]:
    """Tune the official macro F0.5: one equally weighted score per S1 entity."""
    top = _top_with_margin(scored)
    evaluation = target_truth.select("target_key", "true_owner").join(
        top.select(
            "target_key", "source1_entity_id", "probability", "probability_margin"
        ), on="target_key", how="left"
    ).with_columns(
        pl.col("source1_entity_id").fill_null(""),
        pl.col("probability").fill_null(0.0),
        pl.col("probability_margin").fill_null(0.0),
    )
    actual_positive = evaluation.filter(pl.col("true_owner") != "").height
    probabilities = evaluation["probability"].to_numpy()
    margins = evaluation["probability_margin"].to_numpy()
    predicted_owner = evaluation["source1_entity_id"].to_numpy()
    true_owner = evaluation["true_owner"].to_numpy()
    correct = predicted_owner == true_owner

    # Keep the entity universe fixed across thresholds. This prevents a
    # threshold from looking better merely by removing difficult S1 rows from
    # the macro average. Candidate-only owners represent potential false-positive
    # recipient businesses and therefore must participate in calibration.
    owners = np.union1d(
        true_owner[true_owner != ""], predicted_owner[predicted_owner != ""]
    )
    predicted_owner_index = np.searchsorted(owners, predicted_owner)
    true_present = true_owner != ""
    true_owner_index = np.searchsorted(owners, true_owner[true_present])
    true_counts = np.bincount(true_owner_index, minlength=len(owners))
    threshold_grid = np.unique(np.r_[
        np.linspace(0.20, 0.995, 64),
        np.quantile(probabilities, np.linspace(.35, .9995, 36)),
    ])
    margin_grid = np.unique(np.r_[
        0.0, np.linspace(0.005, 0.20, 18),
        np.quantile(margins, np.linspace(.05, .80, 9)),
    ])
    best = {"threshold": 0.9, "margin": 0.0, "f0_5": -1.0,
            "precision": 0.0, "recall": 0.0}
    for threshold in threshold_grid:
        above_threshold = probabilities >= float(threshold)
        for margin in margin_grid:
            predicted = above_threshold & (margins >= float(margin))
            predicted_count = int(predicted.sum())
            tp = int((predicted & correct).sum())
            fp = predicted_count - tp
            fn = actual_positive - tp
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            pred_counts = np.bincount(
                predicted_owner_index[predicted], minlength=len(owners)
            )
            tp_counts = np.bincount(
                predicted_owner_index[predicted & correct], minlength=len(owners)
            )
            fn_counts = true_counts - tp_counts
            fp_counts = pred_counts - tp_counts
            denominator = 1.25 * tp_counts + 0.25 * fn_counts + fp_counts
            entity_scores = np.divide(
                1.25 * tp_counts,
                denominator,
                out=np.ones(len(owners), dtype=np.float64),
                where=denominator > 0,
            )
            score = float(entity_scores.mean())
            conservative_tie = (
                abs(score - best["f0_5"]) <= 1e-12
                and (threshold + margin) > (best["threshold"] + best["margin"])
            )
            if score > best["f0_5"] + 1e-12 or conservative_tie:
                best = {"threshold": float(threshold), "margin": float(margin),
                        "f0_5": score, "macro_f0_5": score,
                        "precision": precision, "recall": recall}
    return best["threshold"], best["margin"], best
