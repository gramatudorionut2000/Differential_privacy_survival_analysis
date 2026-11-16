from __future__ import annotations

import argparse
import copy

import torch

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


def _val_loss(unit: FitUnit, data, loss_fn, device: torch.device) -> float:
    X, t, y = subset(data, "val", unit.val_mask)
    if len(y) == 0:
        return float("nan")
    unit.module.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        tb = torch.from_numpy(t).to(device)
        yb = torch.from_numpy(y).to(device)
        return float(loss_fn(unit.forward(xb, tb), yb).item())


def train_unit_nonprivate(
    unit: FitUnit, data, cfg: ExperimentConfig, device: torch.device
) -> UnitResult:
    X, t, y = subset(data, "train", unit.train_mask)
    loader = make_loader(X, t, y, cfg.optim.batch_size, shuffle=True)
    optimizer = build_optimizer(cfg, unit.module.parameters())
    loss_fn = make_loss(cfg.optim.pos_weight, device)

    history: list[dict[str, float]] = []
    total_steps = 0
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    patience = cfg.optim.early_stopping_patience

    for epoch in range(cfg.optim.epochs):
        train_loss, steps = run_epoch(unit, loader, optimizer, loss_fn, device)
        total_steps += steps
        record = {"epoch": epoch, "train_loss": train_loss}

        if patience is not None:
            val_loss = _val_loss(unit, data, loss_fn, device)
            record["val_loss"] = val_loss
            history.append(record)
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = copy.deepcopy(unit.module.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        else:
            history.append(record)

    if best_state is not None:
        unit.module.load_state_dict(best_state)

    return UnitResult(
        name=unit.name,
        epsilon=None,
        noise_multiplier=None,
        steps=total_steps,
        n_train=len(y),
        history=history,
    )


def train_nonprivate(
    cfg: ExperimentConfig, data=None, device: torch.device | None = None
) -> TrainResult:
    device = device or get_device(cfg.device)
    set_seed(cfg.seed)
    if data is None:
        data = prepare(cfg.data, seed=cfg.seed)

    model = build_model(cfg.model, data.n_features).to(device)
    units = build_fit_units(model, data)
    unit_results = [train_unit_nonprivate(u, data, cfg, device) for u in units]

    return finalize(model, data, cfg, unit_results, epsilon=None, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a causal learner without DP")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--learner", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted config overrides, e.g. optim.lr=3e-4")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    overrides = parse_overrides(args.set)
    if args.learner:
        overrides["model.learner"] = args.learner
    if args.seed is not None:
        overrides["seed"] = args.seed
    overrides.setdefault("privacy.enabled", False)
    cfg = cfg.copy_with(overrides)

    result = train_nonprivate(cfg)
    print(f"[{cfg.run_id}]")
    for k, v in result.test.to_dict().items():
        print(f"  test_{k:22s} {v:.4f}" if isinstance(v, float) else f"  test_{k:22s} {v}")


if __name__ == "__main__":
    main()
