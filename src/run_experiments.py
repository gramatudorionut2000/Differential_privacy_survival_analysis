from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from .config import ExperimentConfig, parse_overrides
from .data_prep import TwinsData, prepare
from .train_dp import train_dp
from .train_nonprivate import train_nonprivate
from .tune import run_search
from .utils import ensure_dir, get_device


MAX_PLOTTABLE_LEARNERS = 4


def expand(cfg: ExperimentConfig) -> list[ExperimentConfig]:
    """Cross the grid axes into concrete run configs."""
    epsilons: list[float | None] = (
        list(cfg.epsilons) if cfg.privacy.enabled else [None]
    )
    runs: list[ExperimentConfig] = []
    for learner in cfg.learners:
        for eps in epsilons:
            for seed in cfg.seeds:
                overrides = {"model.learner": learner, "seed": seed}
                if eps is not None:
                    overrides["privacy.target_epsilon"] = eps
                runs.append(cfg.copy_with(overrides))
    return runs


def run_one(cfg: ExperimentConfig, data: TwinsData, device: torch.device) -> dict:
    """Execute a single config and return its results row."""
    if cfg.tune.enabled:
        res = run_search(cfg, data=data, device=device, verbose=False)
        row = res.best_result.to_row()
        row["achieved_epsilon"] = res.total_epsilon
        row["quoted_epsilon"] = res.quoted_epsilon
        row["n_trials"] = len(res.trials)
        row["tune_accounting"] = res.accounting
        row["best_trial_index"] = res.extra["best_trial_index"]
        best = res.best_config
        row.update(
            {
                "lr": best.optim.lr,
                "batch_size": best.optim.batch_size,
                "epochs": best.optim.epochs,
                "hidden_dims": str(best.model.hidden_dims),
                "dropout": best.model.dropout,
                "max_grad_norm": best.privacy.max_grad_norm,
            }
        )
        model = res.best_result.model
    else:
        res = train_dp(cfg, data=data, device=device) if cfg.privacy.enabled \
            else train_nonprivate(cfg, data=data, device=device)
        row = res.to_row()
        row["quoted_epsilon"] = row["achieved_epsilon"]
        row["n_trials"] = 1
        row["tune_accounting"] = "none"
        row["best_trial_index"] = 0
        model = res.model

    if cfg.save_checkpoints:
        ckpt_dir = ensure_dir(cfg.checkpoint_dir)
        torch.save(model.state_dict(), ckpt_dir / f"{cfg.run_id}.pt")

    return row


def _check_dataset_match(out_path: Path, incoming: set[str]) -> None:
    if not out_path.exists():
        return
    try:
        prior = pd.read_csv(out_path)
    except Exception:
        return
    if prior.empty:
        return
    if "dataset" not in prior.columns:
        print(f"[warn] {out_path} predates the `dataset` column, so its source cannot "
              f"be verified. If it is not {sorted(incoming)}, use a different --out.")
        return
    existing = set(prior["dataset"].dropna().astype(str))
    mismatched = existing - incoming
    if mismatched:
        raise SystemExit(
            f"{out_path} already holds runs from dataset(s) {sorted(mismatched)}, but this "
            f"grid is {sorted(incoming)}. Results from different datasets must not "
            f"share a file -- pass a different --out, or --overwrite to replace it."
        )


def load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path)["run_id"].astype(str))
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the experiment grid")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arm YAML paths (globs allowed)")
    ap.add_argument("--out", default=None,
                    help="results CSV; defaults to the arms' own results_path")
    ap.add_argument("--learners", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epsilons", nargs="*", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run configs already present in the output CSV")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the runs that would execute, then exit")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.arms:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise SystemExit(f"No arm config matched {pattern!r}")
        paths.extend(matched)

    grid_overrides = parse_overrides(args.set)
    if args.learners:
        grid_overrides["learners"] = args.learners
    if args.seeds:
        grid_overrides["seeds"] = args.seeds
    if args.epsilons:
        grid_overrides["epsilons"] = args.epsilons

    runs: list[ExperimentConfig] = []
    for p in paths:
        arm_cfg = ExperimentConfig.from_yaml(p)
        if grid_overrides:
            arm_cfg = arm_cfg.copy_with(grid_overrides)
        runs.extend(expand(arm_cfg))

    if args.out:
        out_path = Path(args.out)
    else:
        declared = {r.results_path for r in runs}
        if len(declared) > 1:
            raise SystemExit(
                f"The selected arms declare different results_path values: "
                f"{sorted(declared)}. Run them separately, or pass --out."
            )
        out_path = Path(declared.pop())
    resetting = args.overwrite and out_path.exists() and not args.dry_run
    if resetting:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = out_path.with_suffix(f"{out_path.suffix}.{stamp}.bak")
        out_path.replace(backup)
        print(f"--overwrite: previous results moved to {backup}")
    done = set() if args.overwrite else load_existing(out_path)
    pending = [r for r in runs if r.run_id not in done]

    # Arm names and therefore run_ids are shared across datasets on purpose, so
    # the figures work unchanged for either. That makes it possible to append
    # cevae rows onto a twins30 CSV and silently produce a file that is half one
    # dataset and half the other -- refuse instead.
    _check_dataset_match(out_path, {r.data.dataset for r in runs})

    # the figure palette has a # fixed number of validated categorical slots and refuses to cycle hues, so
    # a wider grid trains fine but cannot be plotted as a single chart.
    n_learners = len({r.model.learner for r in runs})
    if n_learners > MAX_PLOTTABLE_LEARNERS:
        print(f"[warn] {n_learners} learners in this grid, but only "
              f"{MAX_PLOTTABLE_LEARNERS} categorical colour slots are validated. "
              "The runs are fine; src.figures will refuse to plot them all, so "
              "plan to facet or fold some into an 'other' group.")

    print(f"{len(runs)} runs in grid, {len(runs) - len(pending)} already done, "
          f"{len(pending)} to execute")
    if args.dry_run:
        for r in pending:
            print(f"  {r.run_id}")
        return
    if not pending:
        return

    device = get_device(pending[0].device)
    print(f"device: {device}")
    ensure_dir(out_path.parent)

    # Data prep depends on the seed and on every field of DataConfig, so key the
    # cache on the whole thing. Keying on a hand-picked subset would silently
    # hand one arm another arm's splits if the two ever differed in test_size or drop_ambiguous_nines.
    cache: dict[str, TwinsData] = {}
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_start = time.time()

    # Read the prior results
    prior = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()

    for i, cfg in enumerate(pending, 1):
        # json.dumps rather than a tuple of items: it stays hashable even if a
        # list- or dict-valued field is ever added to DataConfig.
        key = json.dumps(
            {"seed": cfg.seed, **dataclasses.asdict(cfg.data)}, sort_keys=True, default=str
        )
        if key not in cache:
            cache[key] = prepare(cfg.data, seed=cfg.seed)
        data = cache[key]

        t0 = time.time()
        try:
            row = run_one(cfg, data, device)
        except Exception as exc:  # keep the grid going; report at the end
            failures.append((cfg.run_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(pending)}] {cfg.run_id} FAILED: {exc}")
            traceback.print_exc()
            continue

        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        eps = row.get("achieved_epsilon")
        eps_s = "inf" if eps is None or (isinstance(eps, float) and eps != eps) else f"{eps:.3f}"
        print(f"[{i}/{len(pending)}] {cfg.run_id}  eps={eps_s}  "
              f"AUROC={row['test_auroc']:.4f}  PEHE={row['test_pehe']:.4f}  "
              f"({row['seconds']}s)")

        # Flush after every run so a crash never loses completed work. Rows for
        # run_ids produced in this session replace any earlier copy.
        df = pd.DataFrame(rows)
        if not prior.empty:
            keep = prior[~prior["run_id"].isin(df["run_id"])]
            df = pd.concat([keep, df], ignore_index=True)
        df.to_csv(out_path, index=False)

    print(f"\n{len(rows)} runs completed in {time.time() - t_start:.1f}s -> {out_path}")
    if failures:
        print(f"{len(failures)} failed:")
        for run_id, msg in failures:
            print(f"  {run_id}: {msg}")


if __name__ == "__main__":
    main()
