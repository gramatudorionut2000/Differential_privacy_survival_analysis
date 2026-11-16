"""Load, clean, and prepare the Twins data, and simulate treatment assignment.

Two sources, selected by ``DataConfig.dataset``
-----------------------------------------------
``twins30`` (default, ``data/Twin_Data.csv.gz``)
    11,400 pairs x 32 columns: 30 covariates plus ``outcome(t=0)`` and
    ``outcome(t=1)``. Mortality 17.69% / 16.08%, ATE -0.0161.

``cevae`` (``data/cevae``, Louizos et al. release)
    Three CSVs (X / T / Y), 71,345 pairs filtered to 11,984 by the canonical
    "both twins under 2000g" rule. 50 covariates; the encoded width depends on
    the ``cevae_*`` flags and is reported in ``meta["n_features"]`` Adds birth weight and birth order, the only
    twin-distinguishing covariates available. See :func:`load_cevae` for the
    handling differences (binary outcomes, real NaNs, cyclical month,
    high-cardinality FIPS codes).

Both paths return the same :class:`TwinsData`, so everything downstream is
agnostic to which is in play.

1. **The outcome columns are not binary.** They hold *days until death within
   the first year*, ranging 0-360, with ``9999`` meaning the twin survived.
   9999 is a sentinel for "no event", not for missing data. The binary label is
   ``y = 1[outcome != 9999]``; this yields mortality of 17.69% (t=0) and 16.08%
   (t=1), matching the published figures for this dataset. Treating 9999 as
   missing and dropping it would delete 82% of the rows and invert the label.

2. **Both potential outcomes are observed for every row.**  the true ITE ``y1 - y0`` is known, so PEHE and ATE error
   are exactly computable even though the model only ever trains on the factual
   outcome.

3. **Treatment assignment has to be simulated.** Because both potential outcomes are recorded, there is no
   assignment in the data at all: you must choose which of the two outcomes to
   reveal. Revealing at random gives an RCT; revealing as a function of the covariates gives a confounded
   observational study. Separately, this file has no birth-weight column, so weight
   is unavailable as a covariate

4. **Missing-value codes are heterogeneous.** See :class:`~src.config.DataConfig`
   for the full reasoning; in short, ``8`` is a real "unknown" code in the
   1/2-coded risk-factor block, and the ``9``s elsewhere are likely legitimate values.

5. **Every covariate is pair-level, so the treatment effect is unlearnable.**
   All 30 describe the mother or the pregnancy and are therefore shared by both
   twins; nothing in ``x`` distinguishes twin 0 from twin 1. Consequently
   ``tau(x) = E[y1 - y0 | x]`` is near-constant: an oracle regressor given
   *both* counterfactuals scores R^2 = -0.03 predicting ``tau`` from ``x``
   (worse than a constant), against +0.37 for the outcome level. Any finding
   that no learner beats the zero-effect baseline is a property of this
   covariate set, not of the estimator or of DP. Reproduced in
   ``notebooks/01_explore_twins.ipynb``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import ensure_dir

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #

OUTCOME_COLS = ["outcome(t=0)", "outcome(t=1)"]

#: 1 = yes, 2 = no, 8 = unknown.
BINARY_12_WITH_UNKNOWN = [
    "herpes", "hydra", "hemo", "incervix", "pre4000", "preterm", "rh", "uterine",
]

#: 1 = yes, 2 = no, no unknown code present.
BINARY_12_CLEAN = [
    "anemia", "cardiac", "lung", "diabetes", "chyper", "phyper", "eclamp",
    "renal", "othermr",
]

#: 1 = married, 2 = not.
BINARY_OTHER = ["dmar"]

#: Unordered; one-hot encoded.
CATEGORICAL_COLS = ["pldel", "resstatb", "adequacy"]

#: Numeric / ordinal; standardized.
CONTINUOUS_COLS = [
    "dtotord", "cigar", "drink", "wtgain", "gestat", "dmage", "dmeduc",
    "mpcb", "nprevist",
]

#: Columns where NCHS *sometimes* uses a 9-code for "unknown", but where every
#: observed 9 in this file is in range for the column. Only used when
#: ``DataConfig.drop_ambiguous_nines`` is enabled.
AMBIGUOUS_NINE_COLS = ["dtotord", "cigar", "wtgain", "dmeduc", "mpcb", "nprevist"]


@dataclass
class TwinsData:
    """Model-ready arrays plus the counterfactual ground truth.

    ``y`` is the *factual* outcome the model is allowed to see. ``y0``/``y1``
    are both potential outcomes, used only for evaluation.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    t_train: np.ndarray
    t_val: np.ndarray
    t_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y0_train: np.ndarray
    y0_val: np.ndarray
    y0_test: np.ndarray
    y1_train: np.ndarray
    y1_val: np.ndarray
    y1_test: np.ndarray
    feature_names: list[str]
    meta: dict[str, Any]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def ite(self, split: str) -> np.ndarray:
        """True individual treatment effect ``y1 - y0`` for a split."""
        return getattr(self, f"y1_{split}") - getattr(self, f"y0_{split}")

    def ate(self, split: str) -> float:
        return float(self.ite(split).mean())




def load_raw(path: str | Path) -> pd.DataFrame:

    df = pd.read_csv(path, compression="infer")
    df.columns = [str(c).strip().strip("'\"‘’") for c in df.columns]
    return df


def clean(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split into covariates and outcomes, and resolve the sentinel codes.

    Returns ``(X_raw, outcomes, report)`` where ``outcomes`` has columns
    ``y0``, ``y1`` (binary mortality) and ``days0``, ``days1`` (days to death,
    NaN when the twin survived).
    """
    missing = [c for c in OUTCOME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected outcome columns {missing} not found in {list(df.columns)}")

    report: dict[str, Any] = {"n_rows_raw": int(len(df))}

    raw0 = df[OUTCOME_COLS[0]].astype(float)
    raw1 = df[OUTCOME_COLS[1]].astype(float)
    outcomes = pd.DataFrame(
        {
            "y0": (raw0 != cfg.survived_code).astype(int),
            "y1": (raw1 != cfg.survived_code).astype(int),
            "days0": raw0.where(raw0 != cfg.survived_code),
            "days1": raw1.where(raw1 != cfg.survived_code),
        },
        index=df.index,
    )
    report["mortality_rate_t0"] = float(outcomes["y0"].mean())
    report["mortality_rate_t1"] = float(outcomes["y1"].mean())

    X = df.drop(columns=OUTCOME_COLS).copy()

    unexpected = set(X.columns) - set(
        BINARY_12_WITH_UNKNOWN + BINARY_12_CLEAN + BINARY_OTHER
        + CATEGORICAL_COLS + CONTINUOUS_COLS
    )
    if unexpected:
        raise ValueError(
            f"Covariates not assigned to a column group: {sorted(unexpected)}. "
            "Add them to a group in src/data_prep.py."
        )

    if cfg.drop_ambiguous_nines:
        mask = np.zeros(len(X), dtype=bool)
        for col in AMBIGUOUS_NINE_COLS:
            if col in X.columns:
                mask |= (X[col] == 9).to_numpy()
        report["n_dropped"] = int(mask.sum())
        report["drop_reason"] = f"rows with a 9 in {AMBIGUOUS_NINE_COLS}"
        X, outcomes = X.loc[~mask], outcomes.loc[~mask]
    else:
        report["n_dropped"] = 0
        report["drop_reason"] = "none (9-codes kept as legitimate values)"

    # --- 1/2 -> 1/0, with 8 -> NaN in the block that uses it --------------- #
    unknown_counts: dict[str, int] = {}
    for col in BINARY_12_WITH_UNKNOWN:
        is_unknown = X[col] == cfg.unknown_code_binary
        unknown_counts[col] = int(is_unknown.sum())
        X[col] = X[col].map({1: 1.0, 2: 0.0}).astype(float)  # 8 falls through to NaN
        if cfg.add_missing_indicators and is_unknown.any():
            X[f"{col}_unknown"] = is_unknown.astype(float)

    for col in BINARY_12_CLEAN + BINARY_OTHER:
        X[col] = X[col].map({1: 1.0, 2: 0.0}).astype(float)
        if X[col].isna().any():
            raise ValueError(f"Unexpected non-1/2 values in {col!r}")

    # Same key both sources use: per-column count of values that were missing or
    # unknown before imputation. Here that is the `8` codes; in the cevae source
    # it is genuine NaN. 
    report["missing_counts"] = {c: n for c, n in unknown_counts.items() if n}
    report["n_rows_clean"] = int(len(X))
    return X.reset_index(drop=True), outcomes.reset_index(drop=True), report


def encode(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot the unordered categoricals; leave everything else numeric."""
    present = [c for c in CATEGORICAL_COLS if c in X.columns]
    return pd.get_dummies(X, columns=present, prefix=present, dtype=float)


# --------------------------------------------------------------------------- #
# Treatment simulation
# --------------------------------------------------------------------------- #


def assignment_weights(columns: Sequence[str], assignment_seed: int) -> np.ndarray:
    """One N(0,1) weight per covariate, keyed by column **name**.

    Drawing ``rng.normal(size=d)`` and applying it by position looks equivalent
    but is not: inserting or reordering a single column shifts every weight
    after it onto a different feature, so two configs that differ by one feature
    end up with unrelated confounding. That silently ruins any ablation over the
    feature set -- ``cevae_weight_features``, ``cevae_include_bord``,
    ``add_missing_indicators`` are all such knobs. Measured on the cevae source,
    adding ``wt_gap``/``wt_mean`` positionally moved 138 of 202 weights and
    dropped the propensity correlation between the two configs to 0.30.

    Keying on the name fixes a feature's weight, so adding or removing
    features changes the mechanism only through those features.

    ``hashlib`` rather than ``hash()``: the builtin is salted per process, so a
    positional draw would at least be reproducible while a ``hash()``-keyed one
    would not be.
    """
    w = np.empty(len(columns), dtype=float)
    for i, name in enumerate(columns):
        digest = hashlib.blake2b(
            f"{assignment_seed}::{name}".encode(), digest_size=8
        ).digest()
        w[i] = np.random.default_rng(int.from_bytes(digest, "big")).normal()
    return w


def simulate_treatment(
    X: pd.DataFrame, cfg: DataConfig, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a confounded treatment assignment.

    ``T ~ Bernoulli(sigmoid(s * z + noise + intercept))`` where ``z`` is a
    standardized random linear index over the covariates. Because ``z`` is built
    from the same covariates that drive the outcome, treatment and outcome share
    a common cause and the assignment is genuinely confounded -- ``s = 0`` gives
    a randomised trial, larger ``s`` gives stronger confounding.

    Weights come from ``cfg.assignment_seed`` via :func:`assignment_weights`, so
    every arm and seed shares one assignment *mechanism*; only the
    Bernoulli/noise draw varies with the run ``seed`` (unless
    ``resample_treatment_per_seed`` is off, in which case the assignment is
    identical across seeds).

    Returns ``(t, propensity)``.
    """
    Xv = X.to_numpy(dtype=float)
    # DGP-side standardization: uses full-sample statistics on purpose, since
    # this is the data-generating process, not something the model may learn.
    mu, sd = np.nanmean(Xv, axis=0), np.nanstd(Xv, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = np.nan_to_num((Xv - mu) / sd)

    w = assignment_weights(list(X.columns), cfg.assignment_seed)

    index = Z @ w
    index_sd = index.std()
    index = index / (index_sd if index_sd > 1e-12 else 1.0)

    draw_seed = seed if cfg.resample_treatment_per_seed else cfg.assignment_seed
    rng = np.random.default_rng(draw_seed)
    noise = rng.normal(scale=cfg.assignment_noise_scale, size=len(index))

    logits = cfg.confounding_strength * index + noise + cfg.assignment_intercept
    propensity = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.random(len(propensity)) < propensity).astype(int)
    return t, propensity



CEVAE_CYCLICAL = ["birmon"]
CEVAE_ORDINAL = ["dlivord_min", "dtotord_min"]
CEVAE_HIGH_CARDINALITY = ["brstate", "stoccfipb", "mplbir"]
#: Coarse geography. Which of these survive is set by ``cevae_keep_regions``.
CEVAE_REGION = ["brstate_reg", "stoccfipb_reg", "mplbir_reg"]

CEVAE_NOMINAL = [
    "pldel", "birattnd", "mrace", "frace", "crace", "ormoth", "orfath",
    "brstate", "stoccfipb", "mplbir", "brstate_reg", "stoccfipb_reg", "mplbir_reg",
]

#: 0/1 flags that pass through untouched.
CEVAE_BINARY = [
    "anemia", "cardiac", "lung", "diabetes", "herpes", "hydra", "hemo", "chyper",
    "phyper", "eclamp", "incervix", "pre4000", "preterm", "renal", "rh",
    "uterine", "othermr", "tobacco", "alcohol", "dmar", "csex",
]

#: each is a continuous quantity (age, years of school, cigarettes/day, gestation weeks, visits)
#: pre-binned into levels. Encoded as ordinals under ``cevae_ordinal_encode``.
CEVAE_BINNED_ORDINAL = [
    "mager8", "dfageq", "meduc6", "feduc6", "cigar6", "drink5",
    "gestat10", "nprevistq", "mpre5", "adequacy", "data_year",
]

#: Bookkeeping columns that are not covariates.
CEVAE_DROP_ALWAYS_PREFIX = ("Unnamed", "infant_id")

#: Union of every role list. Checked in both directions on load: a data column
#: with no role is an error and a
#: declared name absent from the data is also an error, so a typo in one of the
#: lists above cannot quietly do nothing. Defined last so it sees them all.
CEVAE_DECLARED_ROLES = frozenset(
    CEVAE_NOMINAL + CEVAE_BINARY + CEVAE_ORDINAL + CEVAE_BINNED_ORDINAL
    + CEVAE_CYCLICAL + ["bord_0", "bord_1"]
)


def load_cevae(cfg: DataConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the three-file CEVAE release and return ``(X, y0, y1, report)``.

    Differences from the twins30 file that the loader has to absorb:

    * Outcomes are already binary (``mort_0`` / ``mort_1``) -- there is no
      days-to-death encoding and no 9999 sentinel.
    * Birth weights live in their own file and define the arms: rows are ordered
      lighter-first, so ``t=0`` is the lighter twin and ``t=1`` the heavier.
    * Missing values are genuine ``NaN`` rather than in-band codes, and they are
      substantial -- father's education is 28% missing.
    * Categoricals are typed by the release; ``birmon`` is cyclical and is
      encoded as sin/cos so December and January are adjacent.
    """
    base = Path(cfg.cevae_dir)
    X = pd.read_csv(base / "twin_pairs_X_3years_samesex.csv")
    T = pd.read_csv(base / "twin_pairs_T_3years_samesex.csv")
    Y = pd.read_csv(base / "twin_pairs_Y_3years_samesex.csv")
    if not (len(X) == len(T) == len(Y)):
        raise ValueError(f"CEVAE files disagree on length: {len(X)}, {len(T)}, {len(Y)}")

    # Second direction of the role check: catch a name that exists in a role
    # list but not in the release
    stale_roles = CEVAE_DECLARED_ROLES - set(X.columns)
    if stale_roles:
        raise ValueError(
            f"Role lists in src/data_prep.py name columns absent from "
            f"{base}: {sorted(stale_roles)}."
        )

    w0 = T["dbirwt_0"].to_numpy(float)
    w1 = T["dbirwt_1"].to_numpy(float)
    report: dict[str, Any] = {"n_rows_raw": int(len(X))}

    keep = (w0 < cfg.cevae_weight_cap) & (w1 < cfg.cevae_weight_cap)
    report["n_dropped"] = int((~keep).sum())
    report["drop_reason"] = f"pairs with either twin >= {cfg.cevae_weight_cap:.0f}g"
    X, w0, w1 = X[keep].reset_index(drop=True), w0[keep], w1[keep]
    y0 = Y.loc[keep, "mort_0"].to_numpy(float)
    y1 = Y.loc[keep, "mort_1"].to_numpy(float)
    report["mortality_rate_t0"] = float(y0.mean())
    report["mortality_rate_t1"] = float(y1.mean())

    X = X.drop(columns=[c for c in X.columns if c.startswith(CEVAE_DROP_ALWAYS_PREFIX)])
    if cfg.cevae_drop_high_cardinality:
        X = X.drop(columns=[c for c in CEVAE_HIGH_CARDINALITY if c in X.columns])
    unknown_regions = set(cfg.cevae_keep_regions) - set(CEVAE_REGION)
    if unknown_regions:
        raise ValueError(
            f"cevae_keep_regions has unknown entries {sorted(unknown_regions)}; "
            f"pick from {CEVAE_REGION}"
        )
    X = X.drop(columns=[c for c in CEVAE_REGION
                        if c in X.columns and c not in cfg.cevae_keep_regions])

    # Every remaining column must have a declared role so an unrecognised
    # column is an error rather than a silent passthrough.
    unassigned = [c for c in X.columns if c not in CEVAE_DECLARED_ROLES]
    if unassigned:
        raise ValueError(
            f"CEVAE columns with no declared role: {sorted(unassigned)}. Add each "
            "to CEVAE_NOMINAL / CEVAE_BINARY / CEVAE_ORDINAL / "
            "CEVAE_BINNED_ORDINAL in src/data_prep.py."
        )
    if not cfg.cevae_include_bord:
        X = X.drop(columns=[c for c in ("bord_0", "bord_1") if c in X.columns])
    elif {"bord_0", "bord_1"} <= set(X.columns):
        # bord_0 + bord_1 == 3 for every row: one twin is first-born and the
        # other second, so the pair carries a single bit. Storing both costs
        # four columns once the missingness indicators are added, and DP-SGD
        # pays noise per parameter for the duplication.
        b0 = X.pop("bord_0")
        X = X.drop(columns=["bord_1"])
        X["bord_lighter_first"] = b0.map({1.0: 1.0, 2.0: 0.0})  # NaN stays NaN

    # Cyclical month -> sin/cos on a 12-month circle.
    for col in CEVAE_CYCLICAL:
        if col in X.columns:
            angle = 2 * np.pi * (X.pop(col).astype(float) - 1) / 12.0
            X[f"{col}_sin"], X[f"{col}_cos"] = np.sin(angle), np.cos(angle)

    # Missingness is informative here (a mother with no recorded father's
    # education is not a random mother), so flag every column that has any,
    # before imputing. 

    if cfg.add_missing_indicators:
        for col in [c for c in X.columns if X[c].isna().any()]:
            X[f"{col}_unknown"] = X[col].isna().astype(float)
    report["missing_counts"] = {
        c: int(n) for c, n in X.isna().sum().items() if n and not c.endswith("_unknown")
    }

    if cfg.cevae_weight_features == "gap":
        X["wt_gap"] = w1 - w0
        X["wt_mean"] = (w1 + w0) / 2.0
    elif cfg.cevae_weight_features == "raw":
        X["dbirwt_0"], X["dbirwt_1"] = w0, w1
    elif cfg.cevae_weight_features != "none":
        raise ValueError(
            f"Unknown cevae_weight_features {cfg.cevae_weight_features!r}; "
            "use 'gap', 'raw' or 'none'"
        )

    # One-hot the declared nominals -- at whatever width they happen to be --
    # plus the binned-continuous columns if they are not being treated as
    # ordinals. Everything else is already numeric and passes through.
    onehot = [c for c in CEVAE_NOMINAL if c in X.columns]
    if not cfg.cevae_ordinal_encode:
        onehot += [c for c in CEVAE_BINNED_ORDINAL if c in X.columns]
    X = pd.get_dummies(X, columns=onehot, prefix=onehot, dummy_na=False, dtype=float)
    X = X.astype(float)

    report["n_rows_clean"] = int(len(X))
    report["n_onehot_columns"] = len(onehot)
    return X.reset_index(drop=True), y0, y1, report


# --------------------------------------------------------------------------- #
# Splitting, imputation, scaling
# --------------------------------------------------------------------------- #


def prepare(cfg: DataConfig, seed: int = 0) -> TwinsData:
    """Full pipeline: load -> clean -> encode -> simulate T -> split -> scale.

    Dispatches on ``cfg.dataset``; everything after loading is shared, so both
    sources yield the same arrays and the rest of the project is agnostic to
    which one is in play.
    """
    from sklearn.model_selection import train_test_split

    if cfg.dataset == "twins30":
        df = load_raw(cfg.raw_path)
        X_raw, outcomes, report = clean(df, cfg)
        X_enc = encode(X_raw)
        y0 = outcomes["y0"].to_numpy(float)
        y1 = outcomes["y1"].to_numpy(float)
    elif cfg.dataset == "cevae":
        X_enc, y0, y1, report = load_cevae(cfg)
    else:
        raise ValueError(f"Unknown data.dataset {cfg.dataset!r}; use 'twins30' or 'cevae'")

    t, propensity = simulate_treatment(X_enc, cfg, seed)
    y = np.where(t == 1, y1, y0)

    feature_names = list(X_enc.columns)
    X = X_enc.to_numpy(dtype=float)

    # Stratify on the (treatment, factual outcome) cell so every split keeps the
    # same treated fraction and the same (rare) positive rate.
    strata = t * 2 + y
    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=seed, stratify=strata
    )
    rel_val = cfg.val_size / (1.0 - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, random_state=seed, stratify=strata[idx_trainval]
    )

    # Impute missing values with the training median, then scale with
    # training statistics. Both are fit on train only to avoid leakage.
    # (twins30: the `8`-coded unknowns. cevae: genuine NaNs, up to 28% of a
    # column, )
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    def _impute(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        nan_mask = np.isnan(out)
        if nan_mask.any():
            out[nan_mask] = np.take(medians, np.where(nan_mask)[1])
        return out

    X_train, X_val, X_test = _impute(X_train), _impute(X_val), _impute(X_test)

    if cfg.standardize:
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        X_train = (X_train - mu) / sd
        X_val = (X_val - mu) / sd
        X_test = (X_test - mu) / sd

    meta = {
        **report,
        "dataset": cfg.dataset,
        "seed": seed,
        "n_features": X_train.shape[1],
        "treated_fraction": float(t.mean()),
        "propensity_mean": float(propensity.mean()),
        "propensity_std": float(propensity.std()),
        "confounding_strength": cfg.confounding_strength,
        "true_ate": float((y1 - y0).mean()),
        "factual_positive_rate": float(y.mean()),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
    }

    return TwinsData(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_test=X_test.astype(np.float32),
        t_train=t[idx_train].astype(np.float32),
        t_val=t[idx_val].astype(np.float32),
        t_test=t[idx_test].astype(np.float32),
        y_train=y[idx_train].astype(np.float32),
        y_val=y[idx_val].astype(np.float32),
        y_test=y[idx_test].astype(np.float32),
        y0_train=y0[idx_train].astype(np.float32),
        y0_val=y0[idx_val].astype(np.float32),
        y0_test=y0[idx_test].astype(np.float32),
        y1_train=y1[idx_train].astype(np.float32),
        y1_val=y1[idx_val].astype(np.float32),
        y1_test=y1[idx_test].astype(np.float32),
        feature_names=feature_names,
        meta=meta,
    )


def save_processed(data: TwinsData, out_dir: str | Path) -> Path:
    """Persist arrays as ``.npz`` plus a JSON of names and metadata.

    The filename carries the dataset as well as the seed.
    """
    out = ensure_dir(out_dir)
    arrays = {
        k: v for k, v in vars(data).items()
        if isinstance(v, np.ndarray)
    }
    stem = f"{data.meta.get('dataset', 'twins')}_seed{data.meta['seed']}"
    npz_path = out / f"{stem}.npz"
    np.savez_compressed(npz_path, **arrays)
    with open(out / f"{stem}_meta.json", "w") as f:
        json.dump({"feature_names": data.feature_names, "meta": data.meta}, f, indent=2)
    return npz_path


def main() -> None:
    """Build and summarise either source.

        python -m src.data_prep --dataset cevae
        python -m src.data_prep --dataset cevae --set cevae_weight_features=none

    The summary reads only keys both loaders emit, so it needs no branch on
    which dataset is in play.
    """
    import argparse
    import dataclasses

    from .config import parse_overrides

    ap = argparse.ArgumentParser(description="Prepare a dataset and print a summary")
    ap.add_argument("--dataset", default="twins30", choices=["twins30", "cevae"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="DataConfig overrides, e.g. cevae_weight_features=none")
    ap.add_argument("--no-save", action="store_true",
                    help="print the summary without writing data/processed/")
    args = ap.parse_args()

    overrides = parse_overrides(args.set)
    valid = {f.name for f in dataclasses.fields(DataConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise SystemExit(
            f"Unknown DataConfig field(s): {sorted(unknown)}. Valid: {sorted(valid)}"
        )
    cfg = dataclasses.replace(DataConfig(), dataset=args.dataset, **overrides)

    data = prepare(cfg, seed=args.seed)
    m = data.meta
    print(f"dataset: {m['dataset']}  (seed {m['seed']})")
    print(f"rows: {m['n_rows_raw']} raw -> {m['n_rows_clean']} clean "
          f"({m['n_dropped']} dropped: {m['drop_reason']})")
    print(f"features: {m['n_features']}  (train/val/test = "
          f"{m['n_train']}/{m['n_val']}/{m['n_test']})")
    print(f"mortality  t=0: {m['mortality_rate_t0']:.4f}   t=1: {m['mortality_rate_t1']:.4f}")
    print(f"true ATE: {m['true_ate']:+.4f}")
    print(f"treated fraction: {m['treated_fraction']:.3f}  "
          f"(propensity {m['propensity_mean']:.3f} +/- {m['propensity_std']:.3f})")
    print(f"factual positive rate: {m['factual_positive_rate']:.4f}")
    miss = m["missing_counts"]
    print(f"columns with missing/unknown values: {len(miss)}")
    for col, n in sorted(miss.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {col:22s} {n:6d}  ({n / m['n_rows_clean']:.1%})")
    if len(miss) > 8:
        print(f"    ... and {len(miss) - 8} more")
    if not args.no_save:
        print(f"saved -> {save_processed(data, cfg.processed_dir)}")


if __name__ == "__main__":
    main()
