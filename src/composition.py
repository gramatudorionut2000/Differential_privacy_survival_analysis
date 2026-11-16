"""Privacy accounting for a hyperparameter search: naive vs advanced vs RDP.

The question this module answers: *if you tune hyperparameters on private data,
what does the search itself cost you?*

Every trial in a search touches the training set, so the total privacy loss is
the composition of all K trial. Reporting only the winner's epsilon is the mistake this module exists
to quantify. Three ways to charge for it, loosest to tightest:

``naive`` (basic / linear composition)
    ``eps_total = K * eps_trial``, ``delta_total = K * delta_trial``. Always
    valid, and badly pessimistic for large K.

``advanced`` (Dwork-Rothblum-Vadhan)
    ``eps_total = sqrt(2K ln(1/delta')) * eps_trial + K * eps_trial * (e^eps_trial - 1)``
    at the cost of an extra ``delta'``. Scales like ``sqrt(K)`` in the leading
    term, so it overtakes naive once K is large or eps_trial is small.

``composed`` (alias: ``rdp``)
    Hand all K runs to the DP accountant as one composition and convert to
    ``(eps, delta)`` once, at the end. Under RDP the Gaussian mechanism composes
    additively and losslessly, which is why this is far tighter than the closed
    forms; it is the honest default.

    Which accountant does the composing is ``PrivacyConfig.accountant`` -- the
    same one Opacus uses during training. That matters: computing the budget
    with RDP while training under PRV would leave the calibrated noise and the
    reported epsilon describing different mechanisms.

The no-tuning arm is the K=1 case: the entire budget goes to one run with fixed
hyperparameters. Comparing it against a K-trial arm *at matched total epsilon*
is the tradeoff the article is about -- tuning buys better hyperparameters but
forces more noise per run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterable, Sequence, Callable
from scipy.optimize import brentq


import numpy as np

from opacus.accountants import create_accountant

#: Accountants that can both train (via Opacus' PrivacyEngine) and compose here.
SUPPORTED_ACCOUNTANTS: tuple[str, ...] = ("rdp", "prv", "gdp")

#: Trial-composition schemes accepted by ``TuneConfig.accounting``.
#: ``rdp`` is kept as a back-compatible alias for ``composed``.
ACCOUNTING_SCHEMES: tuple[str, ...] = ("naive", "advanced", "composed", "rdp")


def normalize_accounting(accounting: str) -> str:
    if accounting not in ACCOUNTING_SCHEMES:
        raise ValueError(
            f"Unknown accounting {accounting!r}; pick from {ACCOUNTING_SCHEMES}"
        )
    return "composed" if accounting == "rdp" else accounting


def check_accountant(accountant: str) -> str:
    if accountant not in SUPPORTED_ACCOUNTANTS:
        raise ValueError(
            f"Unsupported accountant {accountant!r}; pick from {SUPPORTED_ACCOUNTANTS}"
        )
    return accountant


DEFAULT_ORDERS: tuple[float, ...] = (
    tuple(1 + x / 10.0 for x in range(1, 100))
    + tuple(range(12, 128))
    + (128.0, 160.0, 192.0, 256.0, 320.0, 384.0, 512.0, 768.0, 1024.0)
)


# --------------------------------------------------------------------------- #
# Closed-form composition
# --------------------------------------------------------------------------- #


def advanced_composition(
    epsilon: float, delta: float, k: int, delta_prime: float
) -> tuple[float, float]:
    """Dwork-Rothblum-Vadhan advanced composition.

    Returns the ``(eps_total, delta_total)`` of ``k``-fold adaptive composition,
    where ``delta_total = k * delta + delta_prime``.
    """
    if delta_prime <= 0:
        raise ValueError("advanced composition needs delta_prime > 0")
    eps_total = (
        math.sqrt(2 * k * math.log(1 / delta_prime)) * epsilon
        + k * epsilon * math.expm1(epsilon)
    )
    return eps_total, k * delta + delta_prime


# --------------------------------------------------------------------------- #
# Accountant-backed composition for DP-SGD
# --------------------------------------------------------------------------- #


def epsilon_of_shapes(
    shapes: Sequence[tuple[float, int]],
    noise_multiplier: float | Sequence[float],
    delta: float,
    accountant: str = "rdp",
) -> float:
    """Total epsilon of a set of DP-SGD runs, composed by the DP accountant.

    ``shapes`` is one ``(sample_rate, steps)`` per run; ``noise_multiplier`` is
    either a single sigma shared by all of them or one sigma per run. Every run
    is pushed into the accountant's history and converted to ``(eps, delta)``
    exactly once at the end, which is what makes this tighter than converting
    each run separately and summing.

    Routing through ``create_accountant`` rather than calling the RDP analysis
    directly is what keeps this consistent with training: the same accountant
    Opacus uses for the PrivacyEngine also computes the budget here.
    """
    accountant = check_accountant(accountant)
    if not shapes:
        raise ValueError("Need at least one run shape")
    sigmas = (
        [float(noise_multiplier)] * len(shapes)
        if np.isscalar(noise_multiplier)
        else [float(s) for s in noise_multiplier]  # type: ignore[union-attr]
    )
    if len(sigmas) != len(shapes):
        raise ValueError("noise_multiplier must be scalar or one value per shape")

    # Merge runs that share a (sigma, sample_rate): composing identical
    # mechanisms just adds their step counts, so this is exact, and it is how
    # the accountant's own `step()` accumulates. Without it a 64-trial search
    # costs 64 separate curve evaluations per epsilon lookup instead of one
    merged: dict[tuple[float, float], int] = {}
    for sig, (q, steps) in zip(sigmas, shapes):
        merged[(sig, q)] = merged.get((sig, q), 0) + int(steps)

    acct = create_accountant(mechanism=accountant)
    acct.history = [(sig, q, n) for (sig, q), n in merged.items()]
    # `alphas` is RDP-specific; the PRV and GDP accountants take no order grid.
    if accountant == "rdp":
        return float(acct.get_epsilon(delta, alphas=list(DEFAULT_ORDERS)))
    return float(acct.get_epsilon(delta))


def epsilon_of_search(
    noise_multiplier: float,
    sample_rate: float,
    steps_per_trial: int,
    n_trials: int,
    delta: float,
    accountant: str = "rdp",
) -> float:
    """Total epsilon of ``n_trials`` *identical* DP-SGD runs."""
    return epsilon_of_shapes(
        [(sample_rate, steps_per_trial)] * n_trials, noise_multiplier, delta, accountant
    )


# --------------------------------------------------------------------------- #
# Inverting: what noise do I need to afford K trials?
# --------------------------------------------------------------------------- #


def noise_multiplier_for_search(
    total_epsilon: float,
    delta: float,
    sample_rate: float,
    steps_per_trial: int,
    n_trials: int,
    accounting: str = "composed",
    delta_prime: float | None = None,
    tol: float = 1e-3,
    max_sigma: float = 1e5,
    accountant: str = "rdp",
) -> float:
    """Noise multiplier so that the *whole search* fits in ``total_epsilon``.

    The budget is fixed up front and the search is forced to fit inside it, rather than each trial
    quietly spending the full budget.

    ``accounting`` selects how the K trials are charged:

    * ``"composed"`` (alias ``"rdp"``) -- one composition through ``accountant``
    * ``"naive"``    -- each trial gets ``total_epsilon / K``
    * ``"advanced"`` -- invert advanced composition for the per-trial budget
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if total_epsilon <= 0:
        raise ValueError("total_epsilon must be > 0")
    accounting = normalize_accounting(accounting)
    accountant = check_accountant(accountant)

    if accounting == "composed":
        # Monotone in sigma: more noise => smaller epsilon. Bisect.
        return _brent_sigma(
            lambda s: epsilon_of_search(
                s, sample_rate, steps_per_trial, n_trials, delta, accountant
            ),
            total_epsilon,
            tol,
            max_sigma,
        )

    if accounting == "naive":
        eps_trial = total_epsilon / n_trials
        delta_trial = delta / n_trials
    else:  # advanced
        dp = delta_prime if delta_prime is not None else delta / 2
        eps_trial = _invert_advanced(total_epsilon, n_trials, dp)
        delta_trial = max((delta - dp) / n_trials, 1e-12)

    return _brent_sigma(
        lambda s: epsilon_of_search(
            s, sample_rate, steps_per_trial, 1, delta_trial, accountant
        ),
        eps_trial,
        tol,
        max_sigma,
    )


def calibrate_noise_for_trials(
    trial_shapes: Sequence[tuple[float, int]],
    total_epsilon: float,
    delta: float,
    tol: float = 1e-3,
    max_sigma: float = 1e5,
    accountant: str = "rdp",
) -> tuple[float, float]:
    """Single noise multiplier that makes a *heterogeneous* search fit a budget.

    ``noise_multiplier_for_search`` assumes every trial runs the same mechanism.
    That breaks as soon as the search space includes ``batch_size`` or
    ``epochs``, because those change the sampling rate and step count 

    This handles the general case exactly: given each trial's
    ``(sample_rate, steps)``, it bisects one shared sigma such that all trials,
    composed together by ``accountant`` and converted to ``(eps, delta)`` once,
    land on ``total_epsilon``.

    Returns ``(sigma, achieved_epsilon)``.
    """
    if not trial_shapes:
        raise ValueError("Need at least one trial shape")
    accountant = check_accountant(accountant)

    def total_eps(sigma: float) -> float:
        return epsilon_of_shapes(trial_shapes, sigma, delta, accountant)

    sigma = _brent_sigma(total_eps, total_epsilon, tol, max_sigma)
    return sigma, total_eps(sigma)


_SIGMA_FLOOR = 1e-3

def _brent_sigma(
    eps_fn: Callable[[float], float],
    target_epsilon: float,
    tol: float,
    max_sigma: float,
) -> float:
    """Smallest sigma whose epsilon is <= target (epsilon decreases in sigma)."""

    cache: dict[float, float] = {}

    def eps(sigma: float) -> float:
        if sigma not in cache:
            cache[sigma] = eps_fn(sigma)
        return cache[sigma]

    # 1. Bracket: need eps(lo) > target >= eps(hi), or brentq raises.
    lo, hi = _SIGMA_FLOOR, 1.0
    if eps(hi) > target_epsilon:
        while eps(hi) > target_epsilon:
            lo, hi = hi, hi * 2
            if hi > max_sigma:
                raise ValueError(
                    f"Cannot reach epsilon={target_epsilon:g} with "
                    f"sigma <= {max_sigma:g}. Reduce n_trials or epochs, "
                    "or raise the budget."
                )
    else:
        # sigma=1.0 already fits; the root is below it. Walk lo down.
        while eps(lo) <= target_epsilon:
            hi, lo = lo, lo / 8
            if lo < 1e-9:
                return hi                # budget too loose to be meaningful

    root = brentq(
        lambda s: eps(s) - target_epsilon,
        lo,
        hi,
        xtol=tol,
        rtol=8.881784197001252e-16,
    )

    sigma, step = root, tol
    while eps(sigma) > target_epsilon:
        sigma += step
        step *= 2
    return sigma


def _invert_advanced(total_epsilon: float, k: int, delta_prime: float) -> float:
    """Per-trial epsilon such that advanced composition of k trials hits target."""
    lo, hi = 1e-6, 10.0
    while advanced_composition(hi, 0.0, k, delta_prime)[0] < total_epsilon:
        lo = hi
        hi *= 2
        if hi > 1e4:
            break
    for _ in range(200):
        mid = (lo + hi) / 2
        if advanced_composition(mid, 0.0, k, delta_prime)[0] < total_epsilon:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------- #
# T-learner: parallel composition over disjoint arms
# --------------------------------------------------------------------------- #


def t_learner_epsilon(eps0: float, eps1: float, mode: str = "parallel") -> float:
    """Combine the two T-learner sub-model budgets.

    The sub-models are fit on disjoint row sets -- control units train ``net0``,
    treated units train ``net1`` -- so any single record participates in exactly
    one of the two trainings. **Parallel composition** therefore applies and the
    pair costs ``max(eps0, eps1)``.

    ``mode="sequential"`` returns the additive bound instead. That is the
    correct choice if you break disjointness, e.g. by using all rows to pick a
    shared preprocessing step, an early-stopping epoch, or a threshold.
    """
    if mode == "parallel":
        return max(eps0, eps1)
    if mode == "sequential":
        return eps0 + eps1
    raise ValueError(f"Unknown mode {mode!r}; use 'parallel' or 'sequential'")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


@dataclass
class CompositionRow:
    n_trials: int
    accounting: str
    noise_multiplier: float
    eps_per_trial: float
    eps_total: float


def compare_accounting(
    total_epsilon: float,
    delta: float,
    sample_rate: float,
    steps_per_trial: int,
    trial_counts: Iterable[int] = (1, 2, 4, 8, 16, 32, 64),
    accountings: Iterable[str] = ("naive", "advanced", "composed"),
    accountant: str = "rdp",
) -> list[CompositionRow]:
    """Table for the composition figure.

    For each K and each accounting scheme: the noise multiplier needed to keep
    the whole search inside ``total_epsilon``, and the epsilon a single trial
    would report in isolation (the number a careless writeup would quote).
    """
    rows: list[CompositionRow] = []
    for k in trial_counts:
        for acc in accountings:
            try:
                sigma = noise_multiplier_for_search(
                    total_epsilon, delta, sample_rate, steps_per_trial, k, acc,
                    accountant=accountant,
                )
            except ValueError:
                continue
            eps_trial = epsilon_of_search(
                sigma, sample_rate, steps_per_trial, 1, delta, accountant
            )
            rows.append(
                CompositionRow(
                    n_trials=k,
                    accounting=acc,
                    noise_multiplier=sigma,
                    eps_per_trial=eps_trial,
                    eps_total=total_epsilon,
                )
            )
    return rows


def main() -> None:
    import pandas as pd

    from .utils import sampling_shape


    n_train = 6840
    delta = 1.0 / (2 * n_train)
    q, steps = sampling_shape(n_train, 256, 30)
    rows = compare_accounting(
        total_epsilon=1.0, delta=delta, sample_rate=q, steps_per_trial=steps
    )
    df = pd.DataFrame([vars(r) for r in rows])
    print(f"Total budget eps=1.0, delta={delta:.3g}, n={n_train}, batch 256, 30 epochs\n")
    print(df.pivot(index="n_trials", columns="accounting",
                   values="noise_multiplier").to_string(float_format="%.3f"))
    print("\n(noise multiplier required to keep the entire K-trial search within eps=1)")


if __name__ == "__main__":
    main()
