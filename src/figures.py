from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir, sampling_shape

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

#: Categorical slots
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


EPS_RAMP = ["#86b6ef", "#2a78d6", "#184f95"]


def series_color(i: int) -> str:
    """Categorical slot ``i``, in fixed order.

    """
    if i >= len(SERIES):
        raise ValueError(
            f"Only {len(SERIES)} categorical slots are validated; got series {i + 1}. "
            "Fold the extras into an 'other' group or split into small multiples "
            "rather than cycling hues."
        )
    return SERIES[i]


def ramp_color(i: int, n: int) -> str:
    """Ordinal ramp step for item ``i`` of ``n`` (light = low, dark = high)."""
    if n > len(EPS_RAMP):
        raise ValueError(
            f"The ordinal ramp has {len(EPS_RAMP)} validated steps; got {n}. "
            "Add steps from the sequential scale and re-check them against the "
            "surface before using more."
        )
    return EPS_RAMP[i]


LEARNER_LABEL = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "tarnet": "TARNet",
    "baseline": "Baseline MLP",
}
ARM_LABEL = {
    "nonprivate_notune": "Non-private, no tuning",
    "nonprivate_tuned": "Non-private, tuned",
    "dp_notune": "DP, no tuning",
    "dp_tuned": "DP, tuned (privacy-aware)",
    "dp_tuned_naive": "DP, tuned (naive accounting)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.dpi": 130,
    })


def tidy(ax, ylabel: str = "", xlabel: str = "", title: str = "") -> None:
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, color=INK, loc="left", pad=10)


def log_eps_axis(ax, values) -> None:
    ticks = sorted(values)
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(ticks[0] * 0.85, ticks[-1] * 1.45)


def place_end_labels(ax, items: list[tuple[float, float, str]], min_gap: float = 0.055) -> None:
    """Label series at their right-hand end, nudged apart so they never overprint.

    Converging series (S-learner and TARNet meet at large ε) would otherwise
    stack two labels on the same pixel row. Labels are pushed apart to at least
    ``min_gap`` of the axis height, and the whole stack slides down if it would
    overflow the top. Call *after* the axis limits are final.
    """
    if not items:
        return
    lo, hi = ax.get_ylim()
    span = hi - lo
    gap = min_gap * span

    items = sorted(items, key=lambda it: it[1])
    placed: list[float] = []
    for _, y, _ in items:
        placed.append(max(y, placed[-1] + gap) if placed else y)

    overflow = placed[-1] - (hi - 0.02 * span)
    if overflow > 0:
        placed = [p - overflow for p in placed]

    for (x, _, text), y_at in zip(items, placed):
        ax.annotate(
            text, xy=(x, y_at), xytext=(6, 0), textcoords="offset points",
            color=INK_2, fontsize=8, va="center", zorder=4,
        )


def agg(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    """Mean and standard deviation across seeds."""
    g = df.groupby(by, dropna=False)[metric]
    return g.agg(["mean", "std", "count"]).reset_index()


def save(fig, out_dir: Path, name: str) -> Path:
    path = ensure_dir(out_dir) / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _nonprivate_reference(df: pd.DataFrame, learner: str, metric: str) -> float | None:
    ref = df[(df["arm"] == "nonprivate_notune") & (df["learner"] == learner)]
    if ref.empty:
        ref = df[(~df["dp"].astype(bool)) & (df["learner"] == learner)]
    return None if ref.empty else float(ref[metric].mean())


# --------------------------------------------------------------------------- #
# Figure 1 -- privacy/utility tradeoff
# --------------------------------------------------------------------------- #


def fig_privacy_utility(df: pd.DataFrame, out_dir: Path) -> None:
    dp = df[df["dp"].astype(bool) & (df["arm"] == "dp_notune")]
    if dp.empty:
        print("  [skip] figure 1 needs dp_notune rows")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, label in zip(
        axes, ["test_auroc", "test_auprc"], ["Test AUROC", "Test AUPRC"]
    ):
        labels: list[tuple[float, float, str]] = []
        for i, learner in enumerate(sorted(dp["learner"].unique())):
            sub = agg(dp[dp["learner"] == learner], ["target_epsilon"], metric)
            sub = sub.sort_values("target_epsilon")
            color = series_color(i)
            ax.errorbar(
                sub["target_epsilon"], sub["mean"], yerr=sub["std"].fillna(0),
                marker="o", color=color, capsize=3, elinewidth=1.2,
                label=LEARNER_LABEL.get(learner, learner), zorder=3,
            )
            labels.append((
                sub["target_epsilon"].iloc[-1], sub["mean"].iloc[-1],
                LEARNER_LABEL.get(learner, learner),
            ))
            ref = _nonprivate_reference(df, learner, metric)
            if ref is not None:
                ax.axhline(ref, color=color, linestyle=":", linewidth=1.2, alpha=0.55, zorder=2)

        log_eps_axis(ax, dp["target_epsilon"].unique())
        tidy(ax, ylabel=label, xlabel="Privacy budget  ε  (log scale)")
        place_end_labels(ax, labels)

    axes[0].legend(loc="lower right")
    fig.suptitle(
        "Privacy–utility tradeoff: outcome classification degrades as ε tightens",
        color=INK, x=0.005, ha="left", fontsize=12.5,
    )
    fig.text(
        0.005, -0.04,
        "Dotted lines mark each learner's non-private ceiling. Mean ± SD over seeds; DP arm trains once at the full budget.",
        color=MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig1_privacy_utility.png")


# --------------------------------------------------------------------------- #
# Figure 2 -- causal estimation quality
# --------------------------------------------------------------------------- #


def _pehe_verdict(dp: pd.DataFrame) -> str:
    """One line stating what this run's data actually shows.

    Counts learners whose paired gap to the zero-effect baseline has a 95% CI
    excluding zero, in each direction. Written from the numbers rather than
    asserted, so the caption cannot disagree with the chart above it.
    """
    beats, loses = [], []
    for learner in sorted(dp["learner"].unique()):
        if learner == "baseline":
            continue  # zero by construction; it *is* the reference line
        g = dp.loc[dp["learner"] == learner, "pehe_gap"]
        if len(g) < 2:
            continue
        se = g.std(ddof=1) / np.sqrt(len(g))
        lo, hi = g.mean() - 1.96 * se, g.mean() + 1.96 * se
        if hi < 0:
            beats.append(LEARNER_LABEL.get(learner, learner))
        elif lo > 0:
            loses.append(LEARNER_LABEL.get(learner, learner))

    parts = []
    if beats:
        parts.append(f"Beating it (95% CI below 0): {', '.join(beats)}.")
    if loses:
        parts.append(f"Beaten by it: {', '.join(loses)}.")
    if not parts:
        return "No learner is distinguishable from the zero-effect baseline."
    return " ".join(parts) + " Note the effect sizes, not just the signs."


def fig_pehe(df: pd.DataFrame, out_dir: Path) -> None:
    dp = df[df["dp"].astype(bool) & (df["arm"] == "dp_notune")]
    if dp.empty:
        print("  [skip] figure 2 needs dp_notune rows")
        return


    dp = dp.assign(pehe_gap=dp["test_pehe"] - dp["test_pehe_zero_baseline"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, label in zip(
        axes,
        ["pehe_gap", "test_ate_error"],
        ["PEHE − zero-effect baseline\n(paired within seed; <0 is better)", "|ATE error|"],
    ):
        labels: list[tuple[float, float, str]] = []
        for i, learner in enumerate(sorted(dp["learner"].unique())):
            sub = agg(dp[dp["learner"] == learner], ["target_epsilon"], metric).sort_values(
                "target_epsilon"
            )
            color = series_color(i)
            ax.errorbar(
                sub["target_epsilon"], sub["mean"], yerr=sub["std"].fillna(0),
                marker="o", color=color, capsize=3, elinewidth=1.2,
                label=LEARNER_LABEL.get(learner, learner), zorder=3,
            )
            labels.append((
                sub["target_epsilon"].iloc[-1], sub["mean"].iloc[-1],
                LEARNER_LABEL.get(learner, learner),
            ))
        log_eps_axis(ax, dp["target_epsilon"].unique())
        tidy(ax, ylabel=label, xlabel="Privacy budget  ε  (log scale)")
        place_end_labels(ax, labels)


    axes[0].axhline(0.0, color=INK_2, linestyle="--", linewidth=1.3, zorder=2)
    axes[0].annotate(
        "zero-effect baseline  ←  worse above / better below",
        xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
        xytext=(0, 5), textcoords="offset points",
        color=INK_2, fontsize=8.5,
    )

    axes[0].legend(loc="best")
    fig.suptitle(
        "Treatment-effect error, relative to predicting zero effect",
        color=INK, x=0.005, ha="left", fontsize=12.5,
    )
    swing = float(df["test_pehe_zero_baseline"].std())
    fig.text(
        0.005, -0.06,
        f"Paired within seed, so the ~{swing:.3f} between-seed swing in the baseline cancels; "
        f"note the y-scale is thousandths.\n{_pehe_verdict(dp)}",
        color=MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig2_pehe_ate.png")


# --------------------------------------------------------------------------- #
# Figure 3 -- does tuning pay for itself?
# --------------------------------------------------------------------------- #


def fig_tuning_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    arms = ["dp_notune", "dp_tuned"]
    sub = df[df["arm"].isin(arms)]
    if sub["arm"].nunique() < 2:
        print("  [skip] figure 3 needs both dp_notune and dp_tuned rows")
        return

    learners = sorted(sub["learner"].unique())
    epsilons = sorted(sub["target_epsilon"].dropna().unique())

    fig_width = min(3.9 * len(learners), 12.0)
    fig, axes = plt.subplots(
        1, len(learners), figsize=(fig_width, 4.4), sharey=True
    )
    axes = np.atleast_1d(axes)

    width = 0.36
    x = np.arange(len(epsilons))
    for ax, learner in zip(axes, learners):
        for j, arm in enumerate(arms):
            a = agg(
                sub[(sub["learner"] == learner) & (sub["arm"] == arm)],
                ["target_epsilon"], "test_auroc",
            ).sort_values("target_epsilon")
            means = [
                float(a.loc[a["target_epsilon"] == e, "mean"].iloc[0])
                if (a["target_epsilon"] == e).any() else np.nan
                for e in epsilons
            ]
            errs = [
                float(a.loc[a["target_epsilon"] == e, "std"].fillna(0).iloc[0])
                if (a["target_epsilon"] == e).any() else 0.0
                for e in epsilons
            ]
            ax.bar(
                x + (j - 0.5) * width, means, width, yerr=errs, capsize=3,
                color=series_color(j), label=ARM_LABEL[arm], zorder=3,
                edgecolor=SURFACE, linewidth=1.5,
                error_kw={"ecolor": INK_2, "elinewidth": 1.1},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"ε={e:g}" for e in epsilons])
        ax.set_ylim(0.5, None)
        tidy(ax, title=LEARNER_LABEL.get(learner, learner))

    axes[0].set_ylabel("Test AUROC")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "At matched total ε, tuning must out-earn the noise it costs",
        color=INK, x=0.005, ha="left", fontsize=12.5,
    )
    tuned_trials = sub.loc[sub["arm"] == "dp_tuned", "n_trials"]
    k = int(tuned_trials.mode().iloc[0]) if not tuned_trials.empty else 0
    fig.text(
        0.005, -0.04,
        f"Both arms respect the same total ε. The tuned arm splits it across {k} trials, so every trial is noisier.",
        color=MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig3_tuning_tradeoff.png")


# --------------------------------------------------------------------------- #
# Figure 4 -- calibration
# --------------------------------------------------------------------------- #


def _mean_reliability(
    per_seed: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average decile-binned reliability curves across seeds.


    Returns ``(mean_confidence, mean_accuracy, sd_accuracy)``.
    """
    lengths = [len(c["confidence"]) for c in per_seed]
    n_bins = max(set(lengths), key=lengths.count)
    kept = [c for c in per_seed if len(c["confidence"]) == n_bins]
    if len(kept) < len(per_seed):
        print(f"    [note] {len(per_seed) - len(kept)} seed(s) produced a "
              f"different bin count and were excluded from the mean curve")
    conf = np.vstack([c["confidence"] for c in kept])
    acc = np.vstack([c["accuracy"] for c in kept])
    return (
        np.nanmean(conf, axis=0),
        np.nanmean(acc, axis=0),
        np.nanstd(acc, axis=0),
    )


def fig_calibration(
    df: pd.DataFrame, out_dir: Path, learner: str = "tarnet",
    seeds: list[int] | None = None, epsilons: list[float] | None = None,
    nonprivate_config: str = "experiments/arm_nonprivate_notune.yaml",
    dp_config: str = "experiments/arm_dp_notune.yaml", n_bins: int = 10,
) -> None:
    """Reliability curves, averaged over seeds, retrained on the fly.

    Aggregated metrics cannot produce a reliability curve -- it needs the
    per-example predictions -- so this figure retrains the models. It follows
    exactly the protocol the grid uses: the same arm YAMLs
    (``arm_nonprivate_notune`` / ``arm_dp_notune``), the same seed list, the
    same per-seed ``prepare()`` splits, and predictions on each seed's own test
    split. The only difference from a grid run is that the curves are averaged
    rather than reduced to a single ECE number.
    """
    from .config import ExperimentConfig
    from .data_prep import prepare
    from .metrics import calibration_bins
    from .train_dp import train_dp
    from .train_nonprivate import train_nonprivate
    from .trainer import predict
    from .utils import get_device

    np_cfg = ExperimentConfig.from_yaml(nonprivate_config).copy_with(
        {"model.learner": learner}
    )
    dp_cfg = ExperimentConfig.from_yaml(dp_config).copy_with({"model.learner": learner})
    seeds = list(seeds) if seeds else list(dp_cfg.seeds)
    epsilons = list(epsilons) if epsilons else list(dp_cfg.epsilons)
    device = get_device(dp_cfg.device)

  
    data_cache = {s: prepare(dp_cfg.data, seed=s) for s in seeds}

    n_fits = (1 + len(epsilons)) * len(seeds)
    print(f"  training {n_fits} models ({len(seeds)} seeds x "
          f"{1 + len(epsilons)} settings) for the reliability curves...")

    def curves_over_seeds(make_cfg, is_dp: bool) -> list[dict[str, np.ndarray]]:
        out = []
        for s in seeds:
            data = data_cache[s]
            cfg = make_cfg(s)
            res = (train_dp if is_dp else train_nonprivate)(cfg, data=data, device=device)
            p, _ = predict(res.model, data.X_test, data.t_test, device)
            out.append(calibration_bins(data.y_test, p, n_bins=n_bins))
        return out

    curves: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []

    per_seed = curves_over_seeds(lambda s: np_cfg.copy_with({"seed": s}), is_dp=False)
    curves.append(("Non-private", MUTED, *_mean_reliability(per_seed)))

    for i, eps in enumerate(epsilons):
        per_seed = curves_over_seeds(
            lambda s, e=eps: dp_cfg.copy_with({"seed": s, "privacy.target_epsilon": e}),
            is_dp=True,
        )
        curves.append((f"ε = {eps:g}", ramp_color(i, len(epsilons)),
                       *_mean_reliability(per_seed)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate("perfect calibration", xy=(0.62, 0.62), color=MUTED, fontsize=8.5,
                rotation=38, ha="center")
    for label, color, conf, acc, sd in curves:
        m = ~np.isnan(conf) & ~np.isnan(acc)
        ax.fill_between(conf[m], (acc - sd)[m], (acc + sd)[m], color=color,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.plot(conf[m], acc[m], marker="o", color=color, label=label, zorder=3,
                linestyle="--" if label == "Non-private" else "-")
        if m.any():
            ax.annotate(label, xy=(conf[m][-1], acc[m][-1]), xytext=(6, -2),
                        textcoords="offset points", color=INK_2, fontsize=8)
    hi = max(float(np.nanmax(c[3] + c[4])) for c in curves)
    ax.set_xlim(0, min(1.0, hi * 1.35))
    ax.set_ylim(0, min(1.0, hi * 1.35))
    tidy(ax, ylabel="Observed mortality rate", xlabel="Predicted probability",
         title=f"Reliability — {LEARNER_LABEL.get(learner, learner)}, "
               f"mean of {len(seeds)} seeds")
    ax.legend(loc="upper left")

    ax = axes[1]
    dp = df[df["dp"].astype(bool) & (df["arm"] == "dp_notune")] if not df.empty else df
    if not dp.empty:
        ece_labels: list[tuple[float, float, str]] = []
        for i, lrn in enumerate(sorted(dp["learner"].unique())):
            a = agg(dp[dp["learner"] == lrn], ["target_epsilon"], "test_ece").sort_values(
                "target_epsilon"
            )
            ax.errorbar(a["target_epsilon"], a["mean"], yerr=a["std"].fillna(0),
                        marker="o", color=series_color(i), capsize=3,
                        elinewidth=1.2, label=LEARNER_LABEL.get(lrn, lrn), zorder=3)
            ece_labels.append((a["target_epsilon"].iloc[-1], a["mean"].iloc[-1],
                               LEARNER_LABEL.get(lrn, lrn)))
            # Non-private ECE as the reference: the DP curves sit several times
            # above it, which is the actual claim this panel supports.
            ref = _nonprivate_reference(df, lrn, "test_ece")
            if ref is not None:
                ax.axhline(ref, color=series_color(i), linestyle=":",
                           linewidth=1.2, alpha=0.55, zorder=2)
        log_eps_axis(ax, dp["target_epsilon"].unique())
        ax.set_ylim(bottom=0)
        ax.legend(loc="best")
        place_end_labels(ax, ece_labels)
    else:
        ax.text(0.5, 0.5, "no results CSV — run the grid first",
                ha="center", va="center", color=MUTED, transform=ax.transAxes)
    tidy(ax, ylabel="Expected calibration error", xlabel="Privacy budget  ε  (log scale)",
         title="ECE under DP, against the non-private reference (dotted)")

    fig.suptitle(
        "DP-SGD does not just cost accuracy, it costs calibration",
        color=INK, x=0.005, ha="left", fontsize=12.5,
    )
    fig.text(
        0.005, -0.10,
        f"Left: decile bins (equal counts), mean over {len(seeds)} seeds with a ±1 SD band. "
        "Same arm configs and per-seed splits as the grid.\n"
        "DP curves sit far above the diagonal, they underestimate risk severely. "
        "Right: dotted lines are each learner's non-private ECE (DP costs 3.6-6.3x more).",
        color=MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig4_calibration.png")


# --------------------------------------------------------------------------- #
# Figure 5 -- what the search costs
# --------------------------------------------------------------------------- #


def composition_params(dp_config: str) -> dict[str, Any]:
    from .config import ExperimentConfig
    from .data_prep import prepare
    from .train_dp import resolve_delta

    cfg = ExperimentConfig.from_yaml(dp_config)
    n_train = len(prepare(cfg.data, seed=cfg.seeds[0]).y_train)
    return {
        "delta": resolve_delta(cfg, n_train),
        "n_train": n_train,
        "batch_size": cfg.optim.batch_size,
        "epochs": cfg.optim.epochs,
        "accountant": cfg.privacy.accountant,
    }


def fig_composition(
    out_dir: Path, total_epsilon: float = 1.0, delta: float = 1e-5,
    n_train: int = 6840, batch_size: int = 256, epochs: int = 30,
    accountant: str = "rdp",
) -> None:

    from .composition import (
        epsilon_of_search, epsilon_of_shapes, noise_multiplier_for_search,
    )

    q, steps = sampling_shape(n_train, batch_size, epochs)
    ks = [1, 2, 4, 8, 16, 32, 64]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    labels: list[tuple[float, float, str]] = []
    for i, acc in enumerate(["naive", "advanced", "composed"]):
        xs, ys = [], []
        for k in ks:
            try:
                ys.append(noise_multiplier_for_search(
                    total_epsilon, delta, q, steps, k, acc, accountant=accountant))
                xs.append(k)
            except ValueError:
                pass  # budget unreachable at this K under this scheme
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", color=series_color(i), label=acc.capitalize(), zorder=3)
        labels.append((xs[-1], ys[-1], acc.capitalize()))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(right=ks[-1] * 1.9)  # room for the direct labels
    ax.legend(loc="upper left")
    tidy(ax, ylabel="Required noise multiplier  σ", xlabel="Number of tuning trials  K",
         title=f"Noise needed to keep a K-trial search inside ε={total_epsilon:g}")
    place_end_labels(ax, labels)

    ax = axes[1]

    sigma_solo = noise_multiplier_for_search(
        total_epsilon, delta, q, steps, 1, "composed", accountant=accountant
    )
    quoted_solo = epsilon_of_search(sigma_solo, q, steps, 1, delta, accountant)
    quoted, actual = [], []
    for k in ks:
        actual.append(epsilon_of_shapes([(q, steps)] * k, sigma_solo, delta, accountant))
        quoted.append(quoted_solo)
    ax.plot(ks, quoted, marker="o", color=series_color(0), label="Quoted (one trial)", zorder=3)
    ax.plot(ks, actual, marker="o", color=series_color(1), label="Actually spent (all trials)", zorder=3)
    ax.fill_between(ks, quoted, actual, color=series_color(1), alpha=0.10, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(right=ks[-1] * 2.4)  # room for the direct labels
    ax.legend(loc="upper left")
    tidy(ax, ylabel="Privacy budget  ε", xlabel="Number of tuning trials  K",
         title="Naive tuning: the gap between what is reported and what is spent")
    place_end_labels(ax, [(ks[-1], quoted[-1], "Quoted"),
                          (ks[-1], actual[-1], "Actually spent")])

    fig.suptitle(
        "Tuning on private data is not free — the search composes",
        color=INK, x=0.005, ha="left", fontsize=12.5,
    )
    fig.text(
        0.005, -0.04,
        f"δ={delta:.3g}, n={n_train}, batch {batch_size}, {epochs} epochs. "
        "Advanced composition overtakes naive between K=16 and K=32; composed accounting dominates both at every K.",
        color=MUTED, fontsize=8.5, ha="left",
    )
    fig.tight_layout()
    save(fig, out_dir, "fig5_composition.png")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the article figures")
    ap.add_argument("--results", default=None,
                    help="results CSV; defaults to --dp-config's results_path")
    ap.add_argument("--out", default=None,
                    help="output directory; defaults to --dp-config's figures_dir")
    ap.add_argument("--only", nargs="*", type=int, default=None, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--calibration-learner", default="tarnet")
    ap.add_argument("--calibration-seeds", nargs="*", type=int, default=None,
                    help="seeds to average figure 4 over (default: the arm config's seeds)")
    ap.add_argument("--nonprivate-config", default="experiments/arm_nonprivate_notune.yaml",
                    help="arm YAML figure 4 retrains the non-private curve from")
    ap.add_argument("--dp-config", default="experiments/arm_dp_notune.yaml",
                    help="arm YAML figure 4 retrains the DP curves from")
    args = ap.parse_args()

    apply_style()


    _cached: dict[str, Any] = {}

    def arm_cfg():
        if "cfg" not in _cached:
            from .config import ExperimentConfig
            _cached["cfg"] = ExperimentConfig.from_yaml(args.dp_config)
        return _cached["cfg"]

    out_dir = Path(args.out or arm_cfg().figures_dir)
    wanted = set(args.only) if args.only else {1, 2, 3, 4, 5}

    results_path = Path(args.results or arm_cfg().results_path)
    if results_path.exists():
        df = pd.read_csv(results_path)
        print(f"loaded {len(df)} runs from {results_path}")
    else:
        df = pd.DataFrame()
        if wanted - {5}:
            print(f"[warn] {results_path} not found; only figure 5 can be built")

    if 1 in wanted and not df.empty:
        fig_privacy_utility(df, out_dir)
    if 2 in wanted and not df.empty:
        fig_pehe(df, out_dir)
    if 3 in wanted and not df.empty:
        fig_tuning_tradeoff(df, out_dir)
    if 4 in wanted:
        fig_calibration(df, out_dir, args.calibration_learner, args.calibration_seeds,
                        nonprivate_config=args.nonprivate_config,
                        dp_config=args.dp_config)
    if 5 in wanted:
        fig_composition(out_dir, **composition_params(args.dp_config))


if __name__ == "__main__":
    main()
