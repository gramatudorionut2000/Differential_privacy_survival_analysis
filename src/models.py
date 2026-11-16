"""Causal learners in PyTorch: S-learner, T-learner, TARNet, plus a control.

Every learner exposes the same two-method interface so the training loops and
metrics never branch on architecture:

``forward(x, t)``
    Logit for the *factual* outcome, i.e. the outcome under the treatment the
    unit actually received. This is what the loss is computed against.

``predict_po(x)``
    Logits for *both* potential outcomes ``(logit_0, logit_1)``. Used only at
    evaluation time, to form ``tau_hat = sigmoid(logit_1) - sigmoid(logit_0)``.

Opacus compatibility
--------------------

* **No BatchNorm.** It mixes information across a batch, so per-sample
  gradients are undefined. ``ModelConfig.norm`` offers ``"layer"`` (LayerNorm,
  which Opacus supports) or ``"none"``.
* **No data-dependent sub-batching.** TARNet and the T-learner could in
  principle route each row to one head via boolean indexing, but that makes the
  effective batch shape depend on the data and breaks Opacus' grad-sample
  hooks. Instead both heads run on the full batch and the factual logit is
  selected with ``torch.where``. The unselected head simply receives zero
  gradient from that row.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "selu": nn.SELU,
}


def make_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    dropout: float = 0.0,
    activation: str = "relu",
    norm: str = "none",
) -> nn.Sequential:
    """Plain MLP. ``hidden_dims=[]`` gives a bare linear map."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {activation!r}; pick from {sorted(_ACTIVATIONS)}")
    if norm not in ("none", "layer"):
        raise ValueError(
            f"Unknown norm {norm!r}; pick 'none' or 'layer'. BatchNorm is not "
            "supported because it breaks per-sample gradients under DP-SGD."
        )
    act = _ACTIVATIONS[activation]

    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if norm == "layer":
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class CausalLearner(nn.Module):
    """Base class fixing the shared interface."""

    is_multi_model: bool = False

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict_po(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def parts(self) -> list[tuple[str, nn.Module, int]]:
        """``(name, submodule, treatment_arm)`` for multi-model learners.

        Only meaningful when ``is_multi_model`` is set; single-model learners
        are trained as one unit and never have this called.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a single-model learner"
        )


class BaselineMLP(CausalLearner):
    """Treatment-agnostic classifier: predicts Y from X, ignoring T.

    Included as a control. It is a legitimate outcome classifier but its implied
    treatment effect is identically zero, which anchors how much of any PEHE
    improvement is genuinely causal rather than an artefact of the outcome
    model's accuracy.
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.net = make_mlp(
            n_features, cfg.hidden_dims, 1, cfg.dropout, cfg.activation, cfg.norm
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_po(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.net(x).squeeze(-1)
        return logit, logit


class SLearner(CausalLearner):
    """Single model over ``[X, T]``
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.net = make_mlp(
            n_features + 1, cfg.hidden_dims, 1, cfg.dropout, cfg.activation, cfg.norm
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t.unsqueeze(-1)], dim=-1)).squeeze(-1)

    def predict_po(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        ones = torch.ones_like(zeros)
        l0 = self.net(torch.cat([x, zeros], dim=-1)).squeeze(-1)
        l1 = self.net(torch.cat([x, ones], dim=-1)).squeeze(-1)
        return l0, l1


class TLearner(CausalLearner):
    """Two independent outcome models, one per treatment arm.

    Each sub-model is fit on a *disjoint* subpopulation (control rows for
    ``net0``, treated rows for ``net1``), so every record influences exactly one
    of them. Under DP that means parallel composition applies and the pair costs
    ``max(eps_0, eps_1)`` 
    The catch is sample efficiency: each model sees roughly half the data, and
    DP-SGD's noise is what it is regardless, so the per-model signal-to-noise
    ratio is worse than the S-learner's at equal epsilon.
    """

    is_multi_model = True

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        self.net0 = make_mlp(
            n_features, cfg.hidden_dims, 1, cfg.dropout, cfg.activation, cfg.norm
        )
        self.net1 = make_mlp(
            n_features, cfg.hidden_dims, 1, cfg.dropout, cfg.activation, cfg.norm
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        The other learners are optimised through ``forward``, but a T-learner's
        sub-models are fit separately on their own arm's rows, so training calls
        ``net0``/``net1`` directly (see ``trainer.build_fit_units``) and
        evaluation goes through ``predict_po``. This exists so a ``TLearner``
        still behaves like every other :class:`CausalLearner` when called
        directly, and is equivalent to selecting from ``predict_po``.
        """
        l0 = self.net0(x).squeeze(-1)
        l1 = self.net1(x).squeeze(-1)
        return torch.where(t > 0.5, l1, l0)

    def predict_po(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net0(x).squeeze(-1), self.net1(x).squeeze(-1)

    def parts(self) -> list[tuple[str, nn.Module, int]]:
        """``(name, submodule, treatment_arm)`` for the per-arm training loop."""
        return [("net0", self.net0, 0), ("net1", self.net1, 1)]


class TARNet(CausalLearner):
    """Shared representation trunk with one outcome head per arm.
    
    the trunk pools both arms' data (so it uses every record,
    unlike the T-learner) while the separate heads keep the treatment effect
    from being regularised away (unlike the S-learner). Under DP it stays a
    single model and therefore a single budget.
    """

    def __init__(self, n_features: int, cfg: ModelConfig):
        super().__init__()
        if not cfg.hidden_dims:
            raise ValueError("TARNet needs at least one trunk layer in model.hidden_dims")
        self.trunk = make_mlp(
            n_features,
            cfg.hidden_dims[:-1],
            cfg.hidden_dims[-1],
            cfg.dropout,
            cfg.activation,
            cfg.norm,
        )
        self.trunk_act = _ACTIVATIONS[cfg.activation]()
        rep_dim = cfg.hidden_dims[-1]
        self.head0 = make_mlp(rep_dim, cfg.head_dims, 1, cfg.dropout, cfg.activation, cfg.norm)
        self.head1 = make_mlp(rep_dim, cfg.head_dims, 1, cfg.dropout, cfg.activation, cfg.norm)

    def _represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk_act(self.trunk(x))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rep = self._represent(x)
        l0 = self.head0(rep).squeeze(-1)
        l1 = self.head1(rep).squeeze(-1)
        return torch.where(t > 0.5, l1, l0)

    def predict_po(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rep = self._represent(x)
        return self.head0(rep).squeeze(-1), self.head1(rep).squeeze(-1)


_REGISTRY: dict[str, type[CausalLearner]] = {
    "baseline": BaselineMLP,
    "s_learner": SLearner,
    "t_learner": TLearner,
    "tarnet": TARNet,
}


def build_model(cfg: ModelConfig, n_features: int) -> CausalLearner:
    if cfg.learner not in _REGISTRY:
        raise ValueError(f"Unknown learner {cfg.learner!r}; pick from {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.learner](n_features, cfg)
