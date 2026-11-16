"""
**Naive tuning** runs K trials, each calibrated to the target epsilon on its
own, then reports the winning trial's epsilon as the epsilon.

**Privacy-aware tuning** (``accounting="composed"``) fixes the total budget
first and calibrates the noise so that all K trials together fit inside it.
Each trial is noisier than it would be alone. The composing is done by ``privacy.accountant`` -- the same
accountant Opacus trains under -- so the calibrated noise and the reported
epsilon describe the same mechanism.

Against both sits the **no-tuning** arm: K=1, the whole budget on a single run
with fixed defaults. Whether tuning is worth it is then an empirical question,
which is the point of the experiment.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .composition import (
    calibrate_noise_for_trials,
    epsilon_of_search,
    epsilon_of_shapes,
    normalize_accounting,
)
from .config import ExperimentConfig, parse_overrides
from .data_prep import prepare
from .metrics import EvalResult
from .train_dp import resolve_delta, train_dp
from .train_nonprivate import train_nonprivate
from .trainer import TrainResult
from .utils import get_device, sampling_shape, set_seed


@dataclass
class TrialRecord:
    index: int
    overrides: dict[str, Any]
    val_metric: float
    test_metrics: dict[str, float]
    epsilon: float | None
    noise_multiplier: float | None


@dataclass
class TuneResult:
    best_config: ExperimentConfig
    best_result: TrainResult
    trials: list[TrialRecord]
    #: Epsilon the whole search spent.
    total_epsilon: float | None
    #: Epsilon a single trial reports in isolation.
    quoted_epsilon: float | None
    accounting: str
    noise_multiplier: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

def sample_configs(
    cfg: ExperimentConfig, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Draw ``n_trials`` hyperparameter settings by random search.
    """
    space = cfg.tune.search_space
    if not space:
        raise ValueError("tune.search_space is empty")

    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(cfg.tune.n_trials):
        for _attempt in range(50):
            draw = {k: copy.deepcopy(v[int(rng.integers(len(v)))]) for k, v in space.items()}
            key = repr(sorted(draw.items(), key=lambda kv: kv[0]))
            if key not in seen:
                seen.add(key)
                break
        trials.append(draw)
    return trials


def _binding_noise_multiplier(result: TrainResult) -> float | None:
    """Sigma of the sub-model that costs the most privacy.

    Single-model learners have exactly one. The T-learner has two, fit on
    disjoint arms, and under parallel composition the pair costs the *larger*
    epsilon -- so the unit that binds is the one with the larger epsilon, and it
    is that unit's sigma the accounting has to use.
    """
    scored = [u for u in result.units if u.epsilon is not None]
    if not scored:
        return result.units[0].noise_multiplier if result.units else None
    return max(scored, key=lambda u: u.epsilon).noise_multiplier


def _selection_sign(selection_mode: str) -> float:
    """``+1`` to maximise, ``-1`` to minimise -- and an error for anything else.
    """
    if selection_mode == "max":
        return 1.0
    if selection_mode == "min":
        return -1.0
    raise ValueError(
        f"Unknown tune.selection_mode {selection_mode!r}; use 'max' or 'min'"
    )


def _selection_metric_name(selection_metric: str) -> str:
    name = selection_metric.removeprefix("val_")
    valid = set(EvalResult.__dataclass_fields__)
    if name not in valid:
        raise ValueError(
            f"Unknown tune.selection_metric {selection_metric!r}; "
            f"pick one of {sorted(valid)} (optionally prefixed 'val_')"
        )
    return name


def trial_shape(cfg: ExperimentConfig, n_train: int) -> tuple[float, int]:
    """``(sample_rate, total_steps)`` for one trial -- what the accountant needs.

    ``optim.batch_size`` and ``optim.epochs`` are special among hyperparameters:
    they change the privacy *mechanism*, not just the model. Tuning them is
    legal, but it makes the trials heterogeneous, which is why the budget is
    calibrated from a per-trial list of shapes rather than a single one.
    """
    return sampling_shape(n_train, cfg.optim.batch_size, cfg.optim.epochs)


def _effective_n(cfg: ExperimentConfig, data) -> int:
    """Rows the privacy mechanism actually samples from.

    For the T-learner each sub-model sees only its own arm. The arms are
    disjoint, so parallel composition applies and the pair costs
    ``max(eps_0, eps_1)`` -- which means the budget has to be calibrated against
    whichever arm is more expensive, and that is the smaller one.
    """
    if cfg.model.learner != "t_learner":
        return len(data.y_train)
    n0 = int((data.t_train == 0).sum())
    n1 = int((data.t_train == 1).sum())
    return min(n0, n1)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def run_search(
    cfg: ExperimentConfig, data=None, device: torch.device | None = None, verbose: bool = True
) -> TuneResult:
    """Run the hyperparameter search described by ``cfg.tune``."""
    device = device or get_device(cfg.device)
    set_seed(cfg.seed)
    if data is None:
        data = prepare(cfg.data, seed=cfg.seed)

    rng = np.random.default_rng(cfg.seed)
    draws = sample_configs(cfg, rng)
    trial_cfgs = [cfg.copy_with(d) for d in draws]

    sigma: float | None = None
    achieved_total: float | None = None
    quoted: float | None = None
    delta = resolve_delta(cfg, len(data.y_train))

    if cfg.privacy.enabled:
        if cfg.privacy.target_epsilon is None:
            raise ValueError("A DP tuning arm needs privacy.target_epsilon")
        accountant = cfg.privacy.accountant
        accounting = normalize_accounting(cfg.tune.accounting)
        n_eff = _effective_n(cfg, data)
        shapes = [trial_shape(tc, n_eff) for tc in trial_cfgs]

        if accounting == "naive":
            # Each trial calibrated to the full budget on its own.
            # Opacus does the per-trial calibration inside train_dp, so the true
            # spend is computed after the loop from the sigmas it actually
            # chose
            quoted = cfg.privacy.target_epsilon
        else:
            sigma, achieved_total = calibrate_noise_for_trials(
                shapes, cfg.privacy.target_epsilon, delta, accountant=accountant
            )
            if verbose:
                print(
                    f"  budget eps={cfg.privacy.target_epsilon:g} over "
                    f"{len(shapes)} trials -> sigma={sigma:.3f}"
                )

    records: list[TrialRecord] = []
    best_idx, best_score, best_result = -1, None, None
    sign = _selection_sign(cfg.tune.selection_mode)

    metric_name = _selection_metric_name(cfg.tune.selection_metric)

    for i, tcfg in enumerate(trial_cfgs):
        if cfg.privacy.enabled:
            result = train_dp(tcfg, data=data, device=device, noise_multiplier=sigma)
        else:
            result = train_nonprivate(tcfg, data=data, device=device)

        score = getattr(result.val, metric_name)
        records.append(
            TrialRecord(
                index=i,
                overrides=draws[i],
                val_metric=float(score),
                test_metrics=result.test.to_dict(),
                epsilon=result.epsilon,
                noise_multiplier=_binding_noise_multiplier(result),
            )
        )
        if verbose:
            print(f"  trial {i + 1}/{len(trial_cfgs)}  "
                  f"{cfg.tune.selection_metric}={score:.4f}  {draws[i]}")

        if np.isnan(score):
            continue
        if best_score is None or sign * score > sign * best_score:
            best_idx, best_score, best_result = i, float(score), result

    if best_result is None:
        raise RuntimeError("Every trial produced a NaN selection metric")

    if cfg.privacy.enabled:
        if accounting == "naive":
            missing = [r.index for r in records if r.noise_multiplier is None]
            if missing:
                # Never substitute NaN here: a NaN sigma makes compute_rdp return
                # -inf and sends the accountant's order search into an unbounded
                # loop, so the grid would stall silently instead of failing.
                raise RuntimeError(
                    f"DP trials {missing} reported no noise multiplier, so the "
                    "search's true epsilon cannot be measured."
                )
            sigmas = [r.noise_multiplier for r in records]
            achieved_total = epsilon_of_shapes(shapes, sigmas, delta, accountant)
            
        if sigma is not None:
            q, steps = shapes[best_idx]
            quoted = epsilon_of_search(sigma, q, steps, 1, delta, accountant)

    return TuneResult(
        best_config=trial_cfgs[best_idx],
        best_result=best_result,
        trials=records,
        total_epsilon=achieved_total,
        quoted_epsilon=quoted,
        accounting=normalize_accounting(cfg.tune.accounting),
        noise_multiplier=sigma,
        extra={"best_trial_index": best_idx, "delta": delta},
    )




def main() -> None:
    ap = argparse.ArgumentParser(description="Hyperparameter search, naive or privacy-aware")
    ap.add_argument("--config", default=None)
    ap.add_argument("--learner", default=None)
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--accounting", default=None,
                    choices=["naive", "advanced", "composed", "rdp"],
                    help="how the K trials are charged ('rdp' is an alias for 'composed')")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    overrides = parse_overrides(args.set)
    overrides["tune.enabled"] = True
    if args.learner:
        overrides["model.learner"] = args.learner
    if args.epsilon is not None:
        overrides["privacy.enabled"] = True
        overrides["privacy.target_epsilon"] = args.epsilon
    if args.trials is not None:
        overrides["tune.n_trials"] = args.trials
    if args.accounting:
        overrides["tune.accounting"] = args.accounting
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = cfg.copy_with(overrides)

    res = run_search(cfg)
    print(f"\nbest trial #{res.extra['best_trial_index']}: {res.best_config.model.hidden_dims}, "
          f"lr={res.best_config.optim.lr}")
    if res.total_epsilon is not None:
        print(f"quoted epsilon (one trial):  {res.quoted_epsilon:.4f}")
        print(f"actual epsilon (all trials): {res.total_epsilon:.4f}  [{res.accounting}]")
    print(f"test AUROC={res.best_result.test.auroc:.4f}  "
          f"PEHE={res.best_result.test.pehe:.4f}")


if __name__ == "__main__":
    main()
