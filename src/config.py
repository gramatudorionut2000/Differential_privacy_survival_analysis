"""Config-driven experiment definition.

Everything an experiment needs lives in :class:`ExperimentConfig`. Configs are
written as YAML in ``experiments/`` and composed by inheritance:

    base.yaml           <- shared defaults
      arm_*.yaml        <- one file per arm, `extends: base.yaml`

The runner expands ``arm x epsilon x seed`` into concrete configs, so no arm
needs its own copy of the training code.

This project uses the Twins dataset (for clarity, refered to as twins30) provided on this repository: https://github.com/jzy95310/ICK/tree/main.

References
----------
Jiang, Z., Zheng, T., Liu, Y., & Carlson, D. (2024).
"Incorporating Prior Knowledge into Neural Networks through an
Implicit Composite Kernel."
Transactions on Machine Learning Research.
https://openreview.net/forum?id=HhjSalvWVe

Jiang, Z., Hou, Z., Liu, Y., Ren, Y., Li, K., & Carlson, D. (2023).
"Estimating causal effects using a multi-task deep ensemble."
International Conference on Machine Learning, 15023–15040.

As well as the CEVAE Twins dataset

References
----------

Almond, Douglas, Kenneth Y. Chay, and David S. Lee. "The costs of low birth weight." 
The Quarterly Journal of Economics 120.3 (2005): 1031-1083.

Causal Effect Inference with Deep Latent-Variable Models 
Christos Louizos, Uri Shalit, Joris Mooij, David Sontag, Richard Zemel, Max Welling, 2017

"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml




@dataclass
class DataConfig:
    """How the raw Twins files are processed.

    ``survived_code`` (9999)
        ``outcome(t=0)`` / ``outcome(t=1)`` .
        The outcome columns hold *days until death within the first year*, and 9999
        marks a twin that survived. The binary classification label is
        ``y = 1[outcome != 9999]``, which reproduces the published mortality
        rates for this dataset (17.69% / 16.08%).

    ``unknown_code_binary`` (8)
        The 1/2-coded maternal risk factors (herpes, hydra, hemo, incervix,
        pre4000, preterm, rh, uterine) use 8 = "unknown". These are genuinely
        missing and are imputed, optionally with a missingness indicator.

    ``drop_ambiguous_nines``
        Several NCHS numeric fields use a repunit-9 code for "unknown", but in
        *this* file every 9 is in range for its column (dtotord max 18, cigar
        max 80, wtgain max 98, dmeduc max 17, nprevist max 49, mpcb max 9) and
        no 99/999 codes appear at all. So a 9 here is almost certainly a real
        value and the default is to keep it. Flip this to True only if you want
        the conservative reading; it discards ~1.2k rows of legitimate data.
    """

    #: Which source to build from. ``twins30`` is the 11,400-pair / 30-covariate
    #: file in ``data/``; ``cevae`` is the 11,984-pair / 50-covariate
    #: release in ``data/cevae``, which additionally carries birth weight and
    #: birth order. Both produce an identical :class:`~src.data_prep.TwinsData`,
    #: so everything downstream is unchanged.
    dataset: str = "twins30"

    raw_path: str = "data/Twin_Data.csv.gz"
    processed_dir: str = "data/processed"

    # --- cevae source ----------------------------------------------------- #
    cevae_dir: str = "data/cevae"
    #: Canonical filter: keep pairs where *both* twins are under this weight.
    #: 2000g reproduces the published 11,984 pairs and 18.94% / 16.42% rates.
    cevae_weight_cap: float = 2000.0
    #: Which birth-weight features to expose to the model.
    #:
    #: ``gap``  -- within-pair gap and pair mean. The gap is the only measured
    #:             effect modifier in this dataset: E[tau | gap] runs from
    #:             -0.006 in the closest fifth to -0.069 in the widest
    #: ``none`` -- omit weight entirely; makes the feature set comparable to
    #:             the twins30 file, which has no weight column at all.
    #: ``raw``  -- both raw weights. Diagnostic only: the pair is ordered
    #:             lighter-first, so raw weights make the arm label trivially
    #:             recoverable.
    cevae_weight_features: str = "gap"
    #: Drop brstate / stoccfipb / mplbir. They are nominal FIPS codes at 51-58
    #: levels each, so encoding them honestly costs ~160 mostly-empty one-hot
    #: columns that DP-SGD pays noise for and cannot exploit. Coarse geography
    #: is retained instead via ``cevae_keep_regions``.
    cevae_drop_high_cardinality: bool = True
    #: Keep bord_0/bord_1 (birth order), the only non-weight twin-specific
    #: covariate in the release. Collapsed to one column, since the pair always
    #: sums to 3 and therefore carries a single bit.
    cevae_include_bord: bool = True
    #: Which of the three 9-level region columns to keep.The default keeps birth
    #: state region only -- one 9-column view, at the level the outcome is
    #: plausibly measured.
    cevae_keep_regions: list[str] = field(default_factory=lambda: ["brstate_reg"])
    #: Encode the binned-continuous categoricals (age, education, cigarettes,
    #: gestation, ...) as ordinals rather than one-hot. They are all genuinely
    #: ordered, so a single standardized input is a fair representation and an
    #: MLP can still fit a non-monotone response to it.
    #:
    #: Note the encoding is chosen from the *semantics* of each variable, For the record, the code order is strongly
    #: monotone in risk for meduc6/feduc6/adequacy/data_year and weakly or
    #: non-monotone for gestat10 and cigar6; move those two back to one-hot via
    #: this flag if you want the extra capacity there.
    cevae_ordinal_encode: bool = True

    survived_code: int = 9999
    unknown_code_binary: int = 8
    add_missing_indicators: bool = True
    drop_ambiguous_nines: bool = False

    # --- simulated treatment assignment ---------------------------------- #
    # Treatment assignment is simulated because both potential outcomes are
    # recorded: there is no assignment in the data, only a choice of which
    # outcome to reveal. s = 0 reveals at
    # random (an RCT); s > 0 reveals as a function of x (confounded).
    # T ~ Bernoulli(sigmoid(confounding_strength * w.x + noise + intercept))
    # with w fixed by `assignment_seed` so every arm/seed sees the same
    # assignment mechanism (only the noise draw varies with the run seed).
    confounding_strength: float = 1.0
    assignment_noise_scale: float = 0.1
    assignment_intercept: float = 0.0
    assignment_seed: int = 12345
    resample_treatment_per_seed: bool = True

    test_size: float = 0.2
    val_size: float = 0.2
    standardize: bool = True


@dataclass
class ModelConfig:
    """Which causal learner, and its architecture.

    ``learner`` is one of ``baseline``, ``s_learner``, ``t_learner``, ``tarnet``.
    """

    learner: str = "tarnet"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    # TARNet only: trunk is `hidden_dims`, each outcome head is `head_dims`.
    head_dims: list[int] = field(default_factory=lambda: [64])
    dropout: float = 0.1
    activation: str = "relu"
    # BatchNorm is incompatible with per-sample gradients; "layer" or "none".
    norm: str = "none"


@dataclass
class OptimConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    weight_decay: float = 0.0
    optimizer: str = "adam"
    # Twins mortality is ~17%, so upweight positives in the BCE loss.
    pos_weight: float | None = None
    early_stopping_patience: int | None = None


@dataclass
class PrivacyConfig:
    """DP-SGD settings.

    ``target_epsilon`` is the budget for one training run. What a "run"
    means depends on the arm: in a no-tuning arm it is the whole experiment, in
    a tuning arm it is one trial and the search's total spend is accounted in
    :mod:`src.composition`.

    ``t_learner_budget`` controls how the T-learner's two models are accounted.
    They are fit on disjoint subpopulations (control rows / treated rows), so
    each record influences exactly one model and **parallel composition**
    applies: the pair costs ``max(eps_0, eps_1)``. Set to
    ``"sequential"`` for the conservative additive reading.

    ``accountant`` Opacus trains under it,
    and :mod:`src.composition` composes the tuning trials with it. Keeping them
    the same is the point, as calibrating a budget in RDP while training under
    PRV would leave the noise and the reported epsilon describing different
    mechanisms. One of ``rdp``, ``prv``, ``gdp``.
    """

    enabled: bool = False
    target_epsilon: float | None = None
    # None -> 1 / (2 * n_train),
    target_delta: float | None = None
    max_grad_norm: float = 1.0
    accountant: str = "rdp"
    t_learner_budget: str = "parallel"


@dataclass
class TuneConfig:
    """Hyperparameter search settings.

    ``accounting`` decides how the K trials are charged against the total
    budget when ``PrivacyConfig.enabled`` is set:

    ``naive``     total = K * eps_trial            (linear/basic composition)
    ``advanced``  Dwork-Rothblum-Vadhan advanced composition
    ``composed/rdp``  hand all K runs to ``PrivacyConfig.accountant`` as one
                  composition and convert once
    The no-tuning arm sets ``enabled: false`` and spends the entire budget on a
    single run with fixed defaults, which is the comparison the article makes.
    """

    enabled: bool = False
    n_trials: int = 8
    accounting: str = "composed"
    selection_metric: str = "val_auroc"
    selection_mode: str = "max"
    search_space: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "optim.lr": [3e-4, 1e-3, 3e-3],
            "optim.batch_size": [128, 256, 512],
            "model.hidden_dims": [[64, 32], [128, 64], [256, 128]],
            "model.dropout": [0.0, 0.1, 0.3],
            "privacy.max_grad_norm": [0.5, 1.0, 2.0],
        }
    )


@dataclass
class ExperimentConfig:

    #: Label for the *study*, as distinct from the arm or the dataset. Two runs
    #: can share an arm, a learner, a seed and a dataset and still be different
    #: experiments. Setting ``name`` separates them: it is written to the results CSV and, when it is
    #: not the default, prefixed onto ``run_id`` so the runner does not mistake
    #: one study's rows for the other's.
    name: str = "main"
    arm: str = "nonprivate_notune"
    seed: int = 0
    device: str = "auto"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)


    results_path: str = "results/metrics/results.csv"
    checkpoint_dir: str = "results/checkpoints"
    figures_dir: str = "figures"
    save_checkpoints: bool = True

    #grid axes, consumed by the runner
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    epsilons: list[float] = field(default_factory=lambda: [0.5, 1.0, 3.0])
    learners: list[str] = field(default_factory=lambda: ["s_learner", "t_learner", "tarnet"])

    @property
    def run_id(self) -> str:
        eps = "inf" if not self.privacy.enabled or self.privacy.target_epsilon is None \
            else f"{self.privacy.target_epsilon:g}"
        stem = f"{self.arm}__{self.model.learner}__eps{eps}__seed{self.seed}"
        # Arm names are shared across datasets and studies so the figures work
        # unchanged for any of them. Without these prefixes the run_ids would
        # collide too, and a second study pointed at the first one's CSV would
        # report every run as "already done" and do nothing. Defaults
        # stay unprefixed so existing results keep their ids.
        for prefix, default in ((self.name, "main"), (self.data.dataset, "twins30")):
            if prefix != default:
                stem = f"{prefix}__{stem}"
        return stem

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        return _build(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(load_yaml_with_extends(path))

    def copy_with(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        """Return a deep copy with dotted-path overrides applied.

        >>> cfg.copy_with({"optim.lr": 3e-4, "privacy.target_epsilon": 1.0})
        """
        d = self.to_dict()
        for key, value in overrides.items():
            _set_dotted(d, key, value)
        return ExperimentConfig.from_dict(d)


# --------------------------------------------------------------------------- #
# YAML loading with `extends:` and dotted overrides
# --------------------------------------------------------------------------- #


def load_yaml_with_extends(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, recursively merging any ``extends:`` parent.

    ``extends`` is resolved relative to the child file's directory. Child keys
    win; nested dicts are merged.
    """
    p = Path(path).resolve()
    _seen = _seen or set()
    if p in _seen:
        raise ValueError(f"Circular `extends` chain at {p}")
    _seen.add(p)

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent = load_yaml_with_extends(p.parent / parent_ref, _seen)
    return deep_merge(parent, raw)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = d
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``["optim.lr=3e-4", "privacy.enabled=true"]`` into a dict.

    Values are parsed as YAML, so ``true``, ``3e-4``, ``[64, 32]`` and ``null``
    all come through with the right type.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override {pair!r} is not of the form key=value")
        key, _, raw = pair.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _build(cls: type, d: dict[str, Any]) -> Any:
    """Recursively construct a (possibly nested) dataclass from a dict."""
    if not dataclasses.is_dataclass(cls):
        return d
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        value = d[f.name]
        ftype = hints[f.name]
        # Nested config sections recurse; everything else (scalars, lists, the
        # search-space dict) is taken as-is.
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
