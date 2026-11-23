"""NLL-trained shallow ensemble of readout heads on a frozen MACE body.

DPOSE-style shallow ensemble (Kellner & Ceriotti; Schäfer, Kellner, Kästner,
Ceriotti, "How to Train a Shallow Ensemble",
https://www.alphaxiv.org/abs/2602.15747).  All ``K`` heads share the frozen
foundation descriptors and see the same full training set; they are trained
jointly so that the ensemble mean and spread form a calibrated Gaussian.

For each force component ``c`` (atom x Cartesian direction) of a batch::

    mu_c  = (1/K) sum_k F_{k,c}
    s2_c  = (1/K) sum_k (F_{k,c} - mu_c)^2 + eps          (biased variance)
    NLL   = mean_c 0.5 * ( log s2_c + (y_c - mu_c)^2 / s2_c )

The first ``warmup_fraction`` of the epochs minimise the per-head force MSE
``mean_{k,c} (F_{k,c} - y_c)^2`` instead, so the heads fit the data before the
spread is shaped by the NLL.  Energies are not used (energy weight 0).

MACE's ``run_train`` optimises each head on its own loss, so this module runs
its own loop.  The starting model is a MACE multi-head model produced by
``run_train`` without any optimisation step (same frozen body and random head
initialisation as the MHC arm); the trained model is saved in the same format,
so ``MACECalculator(model_paths=[path], head=name)`` evaluates each head.
"""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from ase import Atoms

from mlpdft.mace_batches import collate, frames_to_data, load_model, model_heads, with_head
from mlpdft.constants import FORCE_KEY

NLL_VARIANCE_EPS = 1e-6  # eV^2/A^2 added to the ensemble variance (sigma >= 1e-3 eV/A)
TRAINABLE_PREFIXES = ("readouts",)


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean over components of 0.5 (log s2 + (y - mu)^2 / s2); no 2*pi constant."""
    return torch.mean(0.5 * (torch.log(variance) + (target - mean) ** 2 / variance))


def ensemble_nll(predictions: torch.Tensor, target: torch.Tensor,
                 eps: float = NLL_VARIANCE_EPS) -> torch.Tensor:
    """NLL of ``target`` under N(mean_k, var_k + eps); ``predictions`` is (K, ...)."""
    mean = predictions.mean(dim=0)
    variance = predictions.var(dim=0, unbiased=False) + eps
    return gaussian_nll(mean, variance, target)


def per_head_mse(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((predictions - target.unsqueeze(0)) ** 2)


def head_forces(model, batch: dict[str, torch.Tensor], n_heads: int,
                training: bool) -> torch.Tensor:
    """Forces of every head, shape (K, n_atoms, 3).

    ``training=True`` keeps the graph (create_graph) so a force loss can be
    back-propagated into the readout weights; the frozen body has
    ``requires_grad=False`` and gets no gradient.
    """
    forces = []
    for head_index in range(n_heads):
        out = model(with_head(batch, head_index), training=training, compute_force=True)
        forces.append(out["forces"])
    return torch.stack(forces, dim=0)


@dataclass
class ShallowConfig:
    epochs: int = 120
    warmup_fraction: float = 0.2
    lr: float = 1e-3
    batch_size: int = 16
    eps: float = NLL_VARIANCE_EPS
    patience: int | None = 50  # NLL-phase epochs without validation improvement
    clip_grad: float | None = 10.0
    eta_min: float = 1e-6
    seed: int = 123
    max_steps: int | None = None  # testing only: stop after this many optimiser steps

    @property
    def warmup_epochs(self) -> int:
        return min(self.epochs, int(round(self.warmup_fraction * self.epochs)))


def _freeze_body(model) -> list[torch.nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        keep = name.startswith(TRAINABLE_PREFIXES)
        parameter.requires_grad_(keep)
        if keep:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("No readout parameters to train")
    return trainable


def _batches(data: list, targets: list[torch.Tensor], batch_size: int,
             order: np.ndarray, device: str):
    for start in range(0, len(order), batch_size):
        chosen = order[start:start + batch_size]
        yield (collate([data[i] for i in chosen], device),
               torch.cat([targets[i] for i in chosen]).to(device))


def _evaluate(model, data, targets, n_heads, batch_size, device, eps) -> dict[str, Any]:
    order = np.arange(len(data))
    nll_sum, sq_sum, var_sum, count = 0.0, 0.0, 0.0, 0
    head_sq = np.zeros(n_heads)
    for batch, target in _batches(data, targets, batch_size, order, device):
        pred = head_forces(model, batch, n_heads, training=False).detach()
        n = target.numel()
        nll_sum += float(ensemble_nll(pred, target, eps)) * n
        sq_sum += float(torch.sum((pred.mean(0) - target) ** 2))
        var_sum += float(torch.sum(pred.var(0, unbiased=False)))
        head_sq += torch.sum((pred - target) ** 2, dim=(1, 2)).cpu().numpy()
        count += n
    return {
        "nll": nll_sum / count,
        "rmse_mean_force": math.sqrt(sq_sum / count),
        "rms_sigma": math.sqrt(var_sum / count),
        "rmse_per_head": np.sqrt(head_sq / count).tolist(),
    }


def train_shallow_ensemble(init_model_path: str | Path, train_frames: Sequence[Atoms],
                           valid_frames: Sequence[Atoms], out_path: str | Path,
                           config: ShallowConfig, device: str = "cpu",
                           dtype: str = "float64", log_path: str | Path | None = None,
                           log=print) -> dict[str, Any]:
    """Jointly train every readout head of a frozen-body MACE model; save it."""
    torch.manual_seed(config.seed)
    model = load_model(init_model_path, device, dtype)
    heads = model_heads(model)
    n_heads = len(heads)
    if n_heads < 2:
        raise ValueError("A shallow ensemble needs at least two heads")
    trainable = _freeze_body(model)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"[shallow-nll] {n_heads} heads {heads}; {n_train:,} trainable of {n_total:,} "
        f"parameters; {config.epochs} epochs ({config.warmup_epochs} MSE warm-up)")

    train_data = frames_to_data(model, train_frames, heads[0])
    valid_data = frames_to_data(model, valid_frames, heads[0])
    as_tensor = lambda frames: [torch.tensor(np.asarray(a.arrays[FORCE_KEY], dtype=float),
                                             dtype=torch.get_default_dtype()) for a in frames]
    train_targets, valid_targets = as_tensor(train_frames), as_tensor(valid_frames)

    optimizer = torch.optim.Adam(trainable, lr=config.lr, amsgrad=True, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs), eta_min=config.eta_min)
    rng = np.random.default_rng(config.seed)
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    best: dict[str, Any] = {"epoch": None, "nll": float("inf"), "state": None}
    stale, steps, history = 0, 0, []
    started = time.perf_counter()
    try:
        for epoch in range(config.epochs):
            phase = "mse" if epoch < config.warmup_epochs else "nll"
            model.train()
            loss_sum, n_batches = 0.0, 0
            order = rng.permutation(len(train_data))
            for batch, target in _batches(train_data, train_targets, config.batch_size, order, device):
                pred = head_forces(model, batch, n_heads, training=True)
                loss = per_head_mse(pred, target) if phase == "mse" else ensemble_nll(pred, target, config.eps)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(trainable, config.clip_grad)
                optimizer.step()
                loss_sum += float(loss.detach())
                n_batches += 1
                steps += 1
                if config.max_steps is not None and steps >= config.max_steps:
                    break
            scheduler.step()
            model.eval()
            metrics = _evaluate(model, valid_data, valid_targets, n_heads,
                                config.batch_size, device, config.eps)
            record = {"epoch": epoch, "phase": phase, "train_loss": loss_sum / max(1, n_batches),
                      "lr": optimizer.param_groups[0]["lr"], **{f"valid_{k}": v for k, v in metrics.items()}}
            history.append(record)
            if log_file:
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            improved = phase == "nll" and metrics["nll"] < best["nll"]
            if improved:
                best = {"epoch": epoch, "nll": metrics["nll"],
                        "state": {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}}
                stale = 0
            elif phase == "nll":
                stale += 1
            log(f"[shallow-nll] epoch {epoch} {phase}: train {record['train_loss']:.4g}, "
                f"valid NLL {metrics['nll']:.4f}, RMSE {metrics['rmse_mean_force']:.4f}, "
                f"rms sigma {metrics['rms_sigma']:.4f}")
            if config.max_steps is not None and steps >= config.max_steps:
                break
            if phase == "nll" and config.patience is not None and stale >= config.patience:
                log(f"[shallow-nll] early stop at epoch {epoch} (patience {config.patience})")
                break
    finally:
        if log_file:
            log_file.close()
    if best["state"] is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in best["state"]:
                    parameter.copy_(best["state"][name])
        selected = best["epoch"]
    else:
        selected = history[-1]["epoch"] if history else None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deepcopy(model).to("cpu"), out_path)
    return {
        "model": str(out_path),
        "heads": heads,
        "config": asdict(config) | {"warmup_epochs": config.warmup_epochs},
        "selected_epoch": selected,
        "selection": "lowest validation NLL among NLL-phase epochs (last epoch if none)",
        "epochs_run": len(history),
        "optimizer_steps": steps,
        "seconds": time.perf_counter() - started,
        "final_valid": history[-1] if history else None,
        "best_valid_nll": best["nll"] if best["state"] is not None else None,
        "trainable_parameters": n_train,
    }
