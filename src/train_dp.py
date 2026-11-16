from __future__ import annotations

import argparse
import warnings

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from .composition import t_learner_epsilon
from .config import ExperimentConfig, parse_overrides
from .data_prep import prepare
from .models import build_model
from .trainer import (
    FitUnit,
    TrainResult,
    UnitResult,
    build_fit_units,
    build_optimizer,
    finalize,
    make_loader,
    make_loss,
    run_epoch,
    subset,
)
from .utils import get_device, set_seed


def resolve_delta(cfg: ExperimentConfig, n_train: int) -> float:
    if cfg.privacy.target_delta is not None:
        return cfg.privacy.target_delta
    return 1.0 / (2 * n_train)


def train_unit_dp(
    unit: FitUnit,
    data,
    cfg: ExperimentConfig,
    device: torch.device,
    delta: float,
    noise_multiplier: float | None = None,
) -> UnitResult:
    X, t, y = subset(data, "train", unit.train_mask)
    if len(y) == 0:
        raise ValueError(f"Fit unit {unit.name!r} has no training rows")

    errors = ModuleValidator.validate(unit.module, strict=False)
    if errors:
        raise ValueError(
            f"Module {unit.name!r} is not DP-compatible: {errors}. "
        )

    loader = make_loader(X, t, y, cfg.optim.batch_size, shuffle=True)
    optimizer = build_optimizer(cfg, unit.module.parameters())
    loss_fn = make_loss(cfg.optim.pos_weight, device)
    engine = PrivacyEngine(accountant=cfg.privacy.accountant)

    common = dict(
        module=unit.module,
        optimizer=optimizer,
        data_loader=loader,
        max_grad_norm=cfg.privacy.max_grad_norm,
        poisson_sampling=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if noise_multiplier is not None:
            wrapped, dp_optimizer, dp_loader = engine.make_private(
                noise_multiplier=noise_multiplier, **common
            )
        else:
            if cfg.privacy.target_epsilon is None:
                raise ValueError(
                    "privacy.enabled is true but neither target_epsilon nor an "
                    "explicit noise_multiplier was supplied"
                )
            wrapped, dp_optimizer, dp_loader = engine.make_private_with_epsilon(
                target_epsilon=cfg.privacy.target_epsilon,
                target_delta=delta,
                epochs=cfg.optim.epochs,
                **common,
            )

    dp_unit = FitUnit(
        name=unit.name,
        module=wrapped,
        train_mask=unit.train_mask,
        val_mask=unit.val_mask,
        uses_treatment=unit.uses_treatment,
    )

    history: list[dict[str, float]] = []
    total_steps = 0
    for epoch in range(cfg.optim.epochs):
        # No val-loss-based early stopping here: every look at the validation
        # split is an unaccounted query against private data. The DP arms train
        # for a fixed number of epochs by design.
        train_loss, steps = run_epoch(dp_unit, dp_loader, dp_optimizer, loss_fn, device)
        total_steps += steps
        history.append({"epoch": epoch, "train_loss": train_loss})

    achieved = float(engine.get_epsilon(delta))
    sigma = float(dp_optimizer.noise_multiplier)

    wrapped.remove_hooks()

    return UnitResult(
        name=unit.name,
        epsilon=achieved,
        noise_multiplier=sigma,
        steps=total_steps,
        n_train=len(y),
        history=history,
    )


def combine_epsilons(cfg: ExperimentConfig, unit_results: list[UnitResult]) -> float:
    """Total epsilon for a learner from its units' epsilons.

    Single-module learners have one unit. The T-learner's two units are fit on
    disjoint subpopulations, so parallel composition applies and the total is
    the max`.
    """
    eps = [u.epsilon for u in unit_results if u.epsilon is not None]
    if not eps:
        return float("nan")
    if len(eps) == 1:
        return eps[0]
    if len(eps) == 2:
        return t_learner_epsilon(eps[0], eps[1], cfg.privacy.t_learner_budget)
    if cfg.privacy.t_learner_budget == "parallel":
        return max(eps)
    return sum(eps)


def train_dp(
    cfg: ExperimentConfig,
    data=None,
    device: torch.device | None = None,
    noise_multiplier: float | None = None,
) -> TrainResult:
    device = device or get_device(cfg.device)
    set_seed(cfg.seed)
    if data is None:
        data = prepare(cfg.data, seed=cfg.seed)

    delta = resolve_delta(cfg, len(data.y_train))
    model = build_model(cfg.model, data.n_features).to(device)
    units = build_fit_units(model, data)
    unit_results = [
        train_unit_dp(u, data, cfg, device, delta, noise_multiplier) for u in units
    ]

    return finalize(
        model, data, cfg, unit_results, epsilon=combine_epsilons(cfg, unit_results),
        device=device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a causal learner with DP-SGD")
    ap.add_argument("--config", default=None)
    ap.add_argument("--learner", default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    overrides = parse_overrides(args.set)
    overrides["privacy.enabled"] = True
    if args.learner:
        overrides["model.learner"] = args.learner
    if args.epsilon is not None:
        overrides["privacy.target_epsilon"] = args.epsilon
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = cfg.copy_with(overrides)

    result = train_dp(cfg)
    print(f"[{cfg.run_id}]  achieved eps = {result.epsilon:.4f}")
    for u in result.units:
        print(f"  unit {u.name:6s} eps={u.epsilon:.4f} sigma={u.noise_multiplier:.4f} "
              f"n={u.n_train} steps={u.steps}")
    for k, v in result.test.to_dict().items():
        print(f"  test_{k:22s} {v:.4f}" if isinstance(v, float) else f"  test_{k:22s} {v}")


if __name__ == "__main__":
    main()
