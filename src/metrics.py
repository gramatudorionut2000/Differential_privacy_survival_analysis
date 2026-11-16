from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)



def auroc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def auprc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, p))


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p))


def cross_entropy(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibration_bins(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict[str, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; use 'quantile' or 'uniform'")

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    n_eff = len(edges) - 1

    counts = np.zeros(n_eff)
    conf = np.full(n_eff, np.nan)
    acc = np.full(n_eff, np.nan)
    for b in range(n_eff):
        m = idx == b
        counts[b] = m.sum()
        if counts[b] > 0:
            conf[b] = p[m].mean()
            acc[b] = y_true[m].mean()
    return {"edges": edges, "counts": counts, "confidence": conf, "accuracy": acc}


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    """Count-weighted mean gap between confidence and realised frequency."""
    b = calibration_bins(y_true, p, n_bins, strategy)
    mask = b["counts"] > 0
    if not mask.any():
        return float("nan")
    gaps = np.abs(b["confidence"][mask] - b["accuracy"][mask])
    weights = b["counts"][mask] / b["counts"][mask].sum()
    return float((gaps * weights).sum())


def maximum_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> float:
    b = calibration_bins(y_true, p, n_bins, strategy)
    mask = b["counts"] > 0
    if not mask.any():
        return float("nan")
    return float(np.max(np.abs(b["confidence"][mask] - b["accuracy"][mask])))


# --------------------------------------------------------------------------- #
# Causal
# --------------------------------------------------------------------------- #


def pehe(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(tau_hat) - np.asarray(tau_true)) ** 2)))


def ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(abs(np.mean(tau_hat) - np.mean(tau_true)))


def policy_risk(tau_hat: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> float:
    """Expected outcome under the treat-if-``tau_hat<0`` policy.

    The outcome is mortality, so a *good* policy treats units with negative
    predicted effect. Lower is better.
    """
    treat = np.asarray(tau_hat) < 0
    return float(np.mean(np.where(treat, y1, y0)))



@dataclass
class EvalResult:
    auroc: float
    auprc: float
    brier: float
    cross_entropy: float
    ece: float
    mce: float
    pehe: float
    ate_error: float
    ate_hat: float
    ate_true: float
    policy_risk: float
    pehe_zero_baseline: float
    positive_rate: float
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    y_true: np.ndarray,
    p_factual: np.ndarray,
    tau_hat: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    n_bins: int = 10,
    calibration_strategy: str = "quantile",
) -> EvalResult:
    y_true = np.asarray(y_true, dtype=float)
    p_factual = np.asarray(p_factual, dtype=float)
    tau_true = np.asarray(y1, dtype=float) - np.asarray(y0, dtype=float)

    return EvalResult(
        auroc=auroc(y_true, p_factual),
        auprc=auprc(y_true, p_factual),
        brier=brier(y_true, p_factual),
        cross_entropy=cross_entropy(y_true, p_factual),
        ece=expected_calibration_error(y_true, p_factual, n_bins, calibration_strategy),
        mce=maximum_calibration_error(y_true, p_factual, n_bins, calibration_strategy),
        pehe=pehe(tau_hat, tau_true),
        ate_error=ate_error(tau_hat, tau_true),
        ate_hat=float(np.mean(tau_hat)),
        ate_true=float(np.mean(tau_true)),
        policy_risk=policy_risk(tau_hat, y0, y1),
        pehe_zero_baseline=float(np.sqrt(np.mean(tau_true**2))),
        positive_rate=float(y_true.mean()),
        n=len(y_true),
    )
