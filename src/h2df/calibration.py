from __future__ import annotations

from typing import Any

import numpy as np


def quantile_edges(values: list[float], bins: int) -> list[float]:
    if bins <= 1 or len(values) < bins:
        return []
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1])).tolist()


def assign_bin(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.asarray(edges), value, side="right"))


def add_calibration_features(
    rows: list[dict[str, Any]],
    length_edges: list[float],
    loss_edges: list[float],
    task: str,
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        enriched = dict(row)
        enriched["length_bin"] = assign_bin(float(row["token_length"]), length_edges)
        enriched["loss_bin"] = assign_bin(float(row["original_loss"]), loss_edges)
        enriched["type"] = row.get("question_type", "all") if task == "tofu" else "all"
        enriched["domain"] = row.get("domain", "default")
        result.append(enriched)
    return result


def matched_scores(
    target: dict[str, Any],
    calibration: list[dict[str, Any]],
    min_bin_size: int,
) -> list[float]:
    """Condition on all features, then progressively back off sparse bins."""
    feature_orders = [
        ("domain", "length_bin", "loss_bin", "type"),
        ("domain", "length_bin", "loss_bin"),
        ("domain", "loss_bin"),
        ("domain",),
        (),
    ]
    for features in feature_orders:
        values = [
            float(row["original_loss"])
            for row in calibration
            if all(row.get(feature) == target.get(feature) for feature in features)
        ]
        if len(values) >= min_bin_size or not features:
            return values
    raise RuntimeError("Calibration set is empty")


def build_targets(
    forget_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    task: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not calibration_rows:
        raise ValueError("Calibration set must not be empty")
    bins = int(config.get("bins", 4))
    length_edges = quantile_edges([row["token_length"] for row in calibration_rows], bins)
    loss_edges = quantile_edges([row["original_loss"] for row in calibration_rows], bins)
    calibration = add_calibration_features(
        calibration_rows, length_edges, loss_edges, task
    )
    forget = add_calibration_features(forget_rows, length_edges, loss_edges, task)
    min_bin_size = int(config.get("min_bin_size", 20))

    targets = []
    matched_sizes: list[int] = []
    for index, row in enumerate(forget):
        scores = matched_scores(row, calibration, min_bin_size)
        matched_sizes.append(len(scores))
        if task == "tofu":
            lower = max(
                float(row["original_loss"]) + float(config.get("delta_min", 0.5)),
                float(np.quantile(scores, float(config.get("rho", 0.7)))),
            )
            upper = None
        else:
            lower = float(np.quantile(scores, float(config.get("rho_low", 0.5))))
            upper = float(np.quantile(scores, float(config.get("rho_high", 0.9))))
        target = {
            "index": index,
            "original_loss": float(row["original_loss"]),
            "lower_target": lower,
        }
        if upper is not None:
            target["upper_target"] = upper
        targets.append(target)

    metadata = {
        "task": task,
        "forget_examples": len(forget),
        "calibration_examples": len(calibration),
        "length_bin_edges": length_edges,
        "original_loss_bin_edges": loss_edges,
        "matched_bin_size": {
            "min": min(matched_sizes),
            "median": float(np.median(matched_sizes)),
            "max": max(matched_sizes),
        },
    }
    return targets, metadata
