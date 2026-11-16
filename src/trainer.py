"""Shared training machinery for the private and non-private loops.

The key abstraction is :class:`FitUnit`. Most learners are one module trained on
all rows with ``forward(x, t)``. The T-learner is two modules, each trained on
its own arm's rows with ``forward(x)``. Expressing both as a list of fit units
means the DP and non-private loops each need one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig
from .data_prep import TwinsData
from .metrics import EvalResult, evaluate
from .models import CausalLearner


@dataclass
class FitUnit:
    """One independently optimised module, with the rows it trains on."""

    name: str
    module: nn.Module
    train_mask: np.ndarray
    val_mask: np.ndarray
    uses_treatment: bool

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.uses_treatment:
            return self.module(x, t)
        return self.module(x).squeeze(-1)


@dataclass
class UnitResult:
    name: str
    epsilon: float | None
    noise_multiplier: float | None
    steps: int
    n_train: int
    history: list[dict[str, float]] = field(default_factory=list)


def _join(values) -> str:
    vals = list(values)
    if all(v is None for v in vals):
        return ""
    return "|".join("" if v is None else f"{v:.6g}" for v in vals)


@dataclass
class TrainResult:
    """Everything one training run produces."""

    model: CausalLearner
    val: EvalResult
    test: EvalResult
    epsilon: float | None
    units: list[UnitResult]
    config: ExperimentConfig

    def to_row(self) -> dict[str, Any]:
        cfg = self.config
        row: dict[str, Any] = {
            "run_id": cfg.run_id,
            "study": cfg.name,
            "dataset": cfg.data.dataset,
            "arm": cfg.arm,
            "learner": cfg.model.learner,
            "seed": cfg.seed,
            "dp": cfg.privacy.enabled,
            "target_epsilon": cfg.privacy.target_epsilon,
            "achieved_epsilon": self.epsilon,
            "tuned": cfg.tune.enabled,
            # One training run is one trial; the grid runner overwrites this
            # with the real count for a tuned arm.
            "n_trials": 1,
            "lr": cfg.optim.lr,
            "batch_size": cfg.optim.batch_size,
            "epochs": cfg.optim.epochs,
            "hidden_dims": str(cfg.model.hidden_dims),
            "dropout": cfg.model.dropout,
            "max_grad_norm": cfg.privacy.max_grad_norm,
            "noise_multipliers": _join(u.noise_multiplier for u in self.units),
            "unit_epsilons": _join(u.epsilon for u in self.units),
            "unit_names": "|".join(u.name for u in self.units),
        }
        row.update({f"test_{k}": v for k, v in self.test.to_dict().items()})
        row.update({f"val_{k}": v for k, v in self.val.to_dict().items()})
        return row




def build_fit_units(model: CausalLearner, data: TwinsData) -> list[FitUnit]:
    if not model.is_multi_model:
        return [
            FitUnit(
                name="model",
                module=model,
                train_mask=np.ones(len(data.y_train), dtype=bool),
                val_mask=np.ones(len(data.y_val), dtype=bool),
                uses_treatment=True,
            )
        ]

    units: list[FitUnit] = []
    for name, submodule, arm in model.parts():
        units.append(
            FitUnit(
                name=name,
                module=submodule,
                train_mask=(data.t_train == arm),
                val_mask=(data.t_val == arm),
                uses_treatment=False,
            )
        )
    return units


def make_loader(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X)),
        torch.from_numpy(np.ascontiguousarray(t)),
        torch.from_numpy(np.ascontiguousarray(y)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def build_optimizer(cfg: ExperimentConfig, params) -> torch.optim.Optimizer:
    name = cfg.optim.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.optim.lr, momentum=0.9, weight_decay=cfg.optim.weight_decay
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optim.optimizer!r}")


def make_loss(pos_weight: float | None, device: torch.device) -> nn.Module:
    """BCE-with-logits, optionally upweighting the ~17% positive class.

    ``pos_weight`` is a config value rather than something computed from the
    labels on purpose, as deriving it from the training set would be an unaccounted
    read of private data.
    """
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pw)




def run_epoch(
    unit: FitUnit,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, int]:
    """Train one epoch. Returns ``(mean_loss, n_steps)``.

    Works unchanged for DP: Opacus swaps in a ``DPOptimizer`` and a Poisson
    sampler
    """
    unit.module.train()
    total, n_steps = 0.0, 0
    for xb, tb, yb in loader:
        if xb.numel() == 0:
            continue
        xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = unit.forward(xb, tb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        n_steps += 1
    return (total / max(n_steps, 1)), n_steps




@torch.no_grad()
def predict(
    model: CausalLearner,
    X: np.ndarray,
    t: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    p_fact, taus = [], []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start : start + batch_size]).to(device)
        tb = torch.from_numpy(t[start : start + batch_size]).to(device)
        l0, l1 = model.predict_po(xb)
        p0, p1 = torch.sigmoid(l0), torch.sigmoid(l1)
        p_fact.append(torch.where(tb > 0.5, p1, p0).cpu().numpy())
        taus.append((p1 - p0).cpu().numpy())
    return np.concatenate(p_fact), np.concatenate(taus)


def evaluate_split(
    model: CausalLearner, data: TwinsData, split: str, device: torch.device
) -> EvalResult:
    X = getattr(data, f"X_{split}")
    t = getattr(data, f"t_{split}")
    y = getattr(data, f"y_{split}")
    y0 = getattr(data, f"y0_{split}")
    y1 = getattr(data, f"y1_{split}")
    p_fact, tau_hat = predict(model, X, t, device)
    return evaluate(y, p_fact, tau_hat, y0, y1)


def finalize(
    model: CausalLearner,
    data: TwinsData,
    cfg: ExperimentConfig,
    units: list[UnitResult],
    epsilon: float | None,
    device: torch.device,
) -> TrainResult:
    return TrainResult(
        model=model,
        val=evaluate_split(model, data, "val", device),
        test=evaluate_split(model, data, "test", device),
        epsilon=epsilon,
        units=units,
        config=cfg,
    )


def subset(data: TwinsData, split: str, mask: np.ndarray) -> tuple[np.ndarray, ...]:
    X = getattr(data, f"X_{split}")[mask]
    t = getattr(data, f"t_{split}")[mask]
    y = getattr(data, f"y_{split}")[mask]
    return X, t, y


