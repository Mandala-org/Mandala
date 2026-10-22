"""Explicit loss, scaling, and optimization policies for configurable models.

Fitted quantities are checkpoint buffers. Statistics are accepted only from an
explicit training subset; validation and test targets never enter this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

import torch
from torch import nn


def masked_error(
    prediction,
    target,
    mask=None,
    *,
    kind="mse",
    complex_reduction="magnitude",
    weights=None,
):
    """Global masked reduction, with an explicit complex-value convention."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shapes differ: {prediction.shape}, {target.shape}"
        )
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    if mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("A boolean mask must have exactly the target shape")
    selected = (prediction - target)[mask]
    if not selected.numel():
        raise ValueError("The objective has no supervised elements")
    if complex_reduction not in {"magnitude", "components"}:
        raise ValueError("complex_reduction must be magnitude or components")
    if selected.is_complex():
        selected = (
            torch.view_as_real(selected)
            if complex_reduction == "components"
            else selected.abs()
        )
    if kind == "mse":
        errors = selected.square()
    elif kind == "mae":
        errors = selected.abs()
    else:
        raise ValueError(f"Unknown objective kind: {kind}")
    if weights is None:
        return errors.mean()
    weights = weights.reshape((-1,) + (1,) * (target.ndim - 1)).expand_as(target)[mask]
    if errors.ndim > weights.ndim:
        weights = weights.unsqueeze(-1).expand_as(errors)
    if (weights < 0).any() or not torch.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Objective weights must be finite, nonnegative and nonempty")
    return (errors * weights).sum() / weights.sum()


class SpeciesIrrepAffine(nn.Module):
    """Species-pair affine output transformation with selectable backward rule.

    ``gradient_mode='identity'`` uses the MACE-H paper's intentional surrogate
    gradient. ``physical`` differentiates the actual affine function and must be
    used for forces, stress, and other derivatives of predicted observables.
    """

    accepts_physical_derivatives = True

    def __init__(
        self,
        num_species,
        irreps,
        *,
        gradient_mode="physical",
        eps=1e-12,
        target="hamiltonian",
        feature_basis=None,
        statistics_basis="yzx",
    ):
        super().__init__()
        from e3nn.o3 import Irreps

        self.irreps = Irreps(irreps)
        if gradient_mode not in {"identity", "physical"}:
            raise ValueError("gradient_mode must be identity or physical")
        self.gradient_mode = gradient_mode
        self.target = target
        self.statistics_coordinates = None
        if feature_basis is not None and feature_basis != statistics_basis:
            from net.transforms.orbital import IrrepCartesianBasis

            self.statistics_coordinates = IrrepCartesianBasis(
                self.irreps, feature_basis, statistics_basis
            )
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(num_species**2, self.irreps.dim))
        self.register_buffer("std", torch.ones(num_species**2, self.irreps.dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.training_manifest_hash = ""
        self.training_ids = ()

    @torch.no_grad()
    def fit(self, training_records: Iterable[Mapping], *, training_ids):
        """Fit source-style per-irrep statistics on the declared training IDs.

        Records contain id, pair_type, coupled, and an optional coupled mask.
        Every record must belong to the exact explicit training manifest.
        """
        ids = tuple(str(x) for x in training_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("training_ids must be a nonempty unique manifest")
        self.training_ids = ids
        allowed = set(ids)
        records = list(training_records)
        if len(records) != len(ids) or {str(r["id"]) for r in records} != allowed:
            raise ValueError(
                "Statistics records must exactly match the training manifest"
            )
        for r in records:
            if str(r.get("split", "train")) != "train":
                raise ValueError("Statistics may only be fitted on training records")
        self.mean.zero_()
        self.std.fill_(1)
        for pair in range(self.mean.shape[0]):
            for (_, ir), sl in zip(self.irreps, self.irreps.slices()):
                values = []
                for r in records:
                    rows = r["pair_type"] == pair
                    x = r["coupled"][rows, sl]
                    mask = r.get("mask")
                    values.append(x.flatten() if mask is None else x[mask[rows, sl]])
                x = torch.cat(values).to(self.mean)
                if x.numel() == 0:
                    continue
                if ir.l == 0:
                    self.mean[pair, sl] = x.mean()
                # Source std uses Bessel's correction. Empty/constant channels
                # safely use unit scale, with no synthetic validation samples.
                scale = x.std(correction=1) if x.numel() > 1 else x.new_tensor(1.0)
                if torch.isfinite(scale) and scale > self.eps:
                    self.std[pair, sl] = scale
        self.fitted.fill_(True)
        self.training_manifest_hash = hashlib.sha256(
            json.dumps(ids).encode()
        ).hexdigest()

    def fit_training_data(self, samples, *, layout, mapper=None, target_frame="global"):
        from data.operator_features import statistics_records

        self.fit(
            statistics_records(
                samples,
                layout,
                target=self.target,
                mapper=mapper,
                target_frame=target_frame,
            ),
            training_ids=[sample["id"] for sample in samples],
        )

    def fitting_provenance(self, *, training_ids=None):
        ids = tuple(training_ids) if training_ids is not None else self.training_ids
        return {
            "target": self.target,
            "training_ids": list(ids),
            "training_manifest_hash": hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest(),
        }

    def get_extra_state(self):
        return {"training_manifest_hash": self.training_manifest_hash}

    def set_extra_state(self, state):
        self.training_manifest_hash = state["training_manifest_hash"]

    def forward(self, x, pair_type, *, physical_derivatives=False):
        if not bool(self.fitted):
            raise RuntimeError(
                "Output statistics must be fitted on training data first"
            )
        mean, std = self.mean[pair_type], self.std[pair_type]
        if self.statistics_coordinates is not None:
            # Statistics are target-representation quantities. Normalize their
            # affine operator at the representation boundary; shared features
            # remain canonical even for non-isotropic imported statistics.
            coordinates = self.statistics_coordinates.matrix.to(x)
            scaled = ((x @ coordinates.T) * std) @ coordinates
            mean = mean @ coordinates
        else:
            scaled = x * std
        if self.gradient_mode == "identity" and not physical_derivatives:
            return x + (scaled - x).detach() + mean
        return scaled + mean


def make_optimizer(parameters, options):
    options = dict(options)
    kind = options.pop("kind", "adam")
    constructors = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "adagrad": torch.optim.Adagrad,
        "rmsprop": torch.optim.RMSprop,
    }
    if kind not in constructors:
        raise ValueError(f"Unknown optimizer: {kind}")
    return constructors[kind](parameters, **options)


class MilestoneDecay(torch.optim.lr_scheduler.LRScheduler):
    """Multiply LR after explicitly numbered zero-based training epochs.

    ``last_epoch`` counts scheduler steps including its initialization step;
    the event for training epoch k therefore occurs at scheduler step k+1.
    Ordinary scheduler state is sufficient for exact segmented continuation.
    """

    def __init__(self, optimizer, milestones, factors, last_epoch=-1):
        import math

        if (
            len(milestones) != len(factors)
            or any(
                isinstance(k, bool) or not isinstance(k, int) or k < 0
                for k in milestones
            )
            or sorted(set(milestones)) != list(milestones)
            or any(not math.isfinite(f) or f <= 0 for f in factors)
        ):
            raise ValueError(
                "Milestones must be increasing unique epochs with positive finite factors"
            )
        self.events = dict(zip(milestones, factors))
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        factor = self.events.get(self.last_epoch - 1, 1.0)
        return [group["lr"] * factor for group in self.optimizer.param_groups]


class SmoothedPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Plateau scheduler on an explicitly checkpointed exponential mean."""

    def __init__(self, optimizer, *, ema_alpha=0.1, **kwargs):
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0,1]")
        self.ema_alpha = ema_alpha
        self.smoothed_metric = None
        super().__init__(optimizer, **kwargs)

    def step(self, metrics, epoch=None):
        value = float(metrics)
        self.smoothed_metric = (
            value
            if self.smoothed_metric is None
            else self.ema_alpha * value + (1 - self.ema_alpha) * self.smoothed_metric
        )
        super().step(self.smoothed_metric, epoch)


def make_scheduler(optimizer, options):
    if options is None:
        return None
    options = dict(options)
    kind = options.pop("kind")
    monitor = options.pop("monitor", "val/loss")
    if kind == "plateau":
        scheduler = SmoothedPlateau(optimizer, **options)
    elif kind == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **options)
    elif kind == "milestone_decay":
        scheduler = MilestoneDecay(optimizer, **options)
    else:
        raise ValueError(f"Unknown scheduler kind {kind}")
    return {"scheduler": scheduler, "monitor": monitor, "interval": "epoch"}
