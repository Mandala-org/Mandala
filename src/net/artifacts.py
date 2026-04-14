from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl

from data.block_matrix import BlockMatrix, IrrepsBlockData
from net.irrep_tools import filter_irreps_block_data_by_irrep, get_all_irreps


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _as_dense(matrix: BlockMatrix) -> torch.Tensor:
    dense = matrix.to_dense()
    return dense.detach().cpu().to(torch.float64)


def _compute_mu_h(H_pred: BlockMatrix, H_gt: BlockMatrix, S: BlockMatrix) -> float:
    numerator = 0.0
    denominator = 0.0
    for key in H_gt.pair_blocks.keys():
        if key not in H_pred.pair_blocks or key not in S.pair_blocks:
            continue
        pred_blocks = H_pred.pair_blocks[key]
        gt_blocks = H_gt.pair_blocks[key]
        s_blocks = S.pair_blocks[key]
        min_n = min(pred_blocks.shape[0], gt_blocks.shape[0], s_blocks.shape[0])
        diff = pred_blocks[:min_n] - gt_blocks[:min_n]
        s_val = s_blocks[:min_n]
        numerator += torch.sum(diff * s_val).item()
        denominator += torch.sum(s_val * s_val).item()
    return numerator / denominator if denominator > 1e-10 else 0.0


def compute_generalized_eigenvalues(H: BlockMatrix, S: BlockMatrix) -> torch.Tensor:
    H_dense = _as_dense(H)
    S_dense = _as_dense(S)
    H_dense = 0.5 * (H_dense + H_dense.T)
    S_dense = 0.5 * (S_dense + S_dense.T)
    L = torch.linalg.cholesky(S_dense)
    tmp = torch.linalg.solve(L, H_dense)
    A = torch.linalg.solve(L, tmp.T).T
    A = 0.5 * (A + A.T)
    return torch.linalg.eigvalsh(A)


def compute_dos_from_eigenvalues(
    eigenvalues: torch.Tensor,
    sigma: float = 0.2,
    bin_width: float = 0.1,
    e_min: float | None = None,
    e_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if e_min is None or e_max is None:
        eig_min = float(torch.min(eigenvalues).item())
        eig_max = float(torch.max(eigenvalues).item())
        span = max(eig_max - eig_min, 1e-6)
        margin = 0.1 * span + 0.05
        e_min = eig_min - margin
        e_max = eig_max + margin

    grid = torch.arange(
        e_min,
        e_max + bin_width,
        bin_width,
        dtype=eigenvalues.dtype,
        device=eigenvalues.device,
    )
    dos = torch.sum(
        torch.exp(-((grid[:, None] - eigenvalues[None, :]) ** 2) / (2 * sigma**2)),
        dim=1,
    ) / (
        torch.sqrt(
            torch.tensor(2 * torch.pi, dtype=eigenvalues.dtype, device=grid.device)
        )
        * sigma
    )
    return grid, dos


def save_dos_comparison_plot(
    H_pred: BlockMatrix,
    H_gt: BlockMatrix,
    S: BlockMatrix,
    output_path: Path | str,
    *,
    sigma: float = 0.2,
    bin_width: float = 0.1,
    title: str = "DOS Comparison",
) -> dict[str, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eig_pred = compute_generalized_eigenvalues(H_pred, S)
    eig_gt = compute_generalized_eigenvalues(H_gt, S)
    abs_err = torch.abs(eig_pred - eig_gt)
    rel_err = abs_err / (torch.abs(eig_gt) + 1e-12)

    eig_min = float(torch.min(torch.min(eig_pred), torch.min(eig_gt)).item())
    eig_max = float(torch.max(torch.max(eig_pred), torch.max(eig_gt)).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = eig_min - margin
    e_max = eig_max + margin

    grid, dos_pred = compute_dos_from_eigenvalues(
        eig_pred, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )
    _, dos_gt = compute_dos_from_eigenvalues(
        eig_gt, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].plot(eig_gt.cpu().numpy(), label="GT", lw=1.5)
    axes[0].plot(eig_pred.cpu().numpy(), label="Pred", lw=1.2)
    axes[0].set_title("Generalized eigenvalues")
    axes[0].legend()
    axes[1].plot(grid.cpu().numpy(), dos_gt.cpu().numpy(), label="GT", lw=1.5)
    axes[1].plot(grid.cpu().numpy(), dos_pred.cpu().numpy(), label="Pred", lw=1.2)
    axes[1].set_title("DOS")
    axes[1].legend()
    axes[2].plot(abs_err.cpu().numpy())
    axes[2].set_title("Abs eig error")
    axes[3].plot(rel_err.cpu().numpy())
    axes[3].set_title("Rel eig error")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    _save_plot(fig, output_path)

    return {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
    }


def _filter_blocks_by_partial_train(
    block_matrix: BlockMatrix, partial_train: str | None
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if partial_train is None:
        return {
            key: (
                blocks,
                torch.ones(blocks.shape[0], dtype=torch.bool, device=blocks.device),
            )
            for key, blocks in block_matrix.pair_blocks.items()
        }

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, blocks in block_matrix.pair_blocks.items():
        edges = block_matrix.pair_edges[key]
        sx, sy, sz, i, j = edges
        is_diag = (i == j) & (sx == 0) & (sy == 0) & (sz == 0)
        is_shifted_self = (i == j) & ~((sx == 0) & (sy == 0) & (sz == 0))
        if partial_train == "diag":
            mask = is_diag
        elif partial_train == "shifted_self":
            mask = is_shifted_self
        elif partial_train == "offdiag":
            mask = i != j
        else:
            raise ValueError(f"Unsupported partial_train={partial_train!r}")
        out[key] = (blocks, mask)
    return out


def compute_distance_error_curve(
    H_pred: BlockMatrix,
    H_gt: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None = None,
    partial_train: str | None = None,
    n_bins: int = 16,
) -> dict[str, Any] | None:
    if positions is None:
        raise ValueError("positions must be provided for distance-binned analysis")

    filtered_pred = _filter_blocks_by_partial_train(H_pred, partial_train)
    filtered_gt = _filter_blocks_by_partial_train(H_gt, partial_train)

    dists = []
    abs_l1_sum = []
    abs_l2_sum = []
    gt_l1_sum = []
    gt_l2_sum = []
    elem_count = []
    edge_l1_abs = []
    edge_l2_abs = []
    edge_l1_rel = []
    edge_l2_rel = []

    for key in H_gt.pair_blocks.keys():
        if key not in H_pred.pair_blocks:
            continue
        pred_blocks_full = H_pred.pair_blocks[key]
        gt_blocks_full = H_gt.pair_blocks[key]
        gt_edges_full = H_gt.pair_edges[key]
        _, pred_mask = filtered_pred[key]
        _, gt_mask = filtered_gt[key]

        pred_edge_to_idx = {
            tuple(map(int, edge.tolist())): idx
            for idx, edge in enumerate(H_pred.pair_edges[key].t())
        }

        for gt_idx, edge in enumerate(gt_edges_full.t()):
            if not bool(gt_mask[gt_idx]):
                continue
            edge_key = tuple(map(int, edge.tolist()))
            pred_idx = pred_edge_to_idx.get(edge_key)
            if pred_idx is None:
                continue
            if (
                pred_idx >= pred_blocks_full.shape[0]
                or gt_idx >= gt_blocks_full.shape[0]
            ):
                continue
            if pred_idx < pred_mask.shape[0] and not bool(pred_mask[pred_idx]):
                continue

            pred_block = pred_blocks_full[pred_idx]
            gt_block = gt_blocks_full[gt_idx]
            diff = pred_block - gt_block

            sx, sy, sz, i, j = edge_key
            shift = torch.tensor(
                [sx, sy, sz], dtype=positions.dtype, device=positions.device
            )
            if box is not None:
                disp = positions[j] - positions[i] + shift @ box
            else:
                disp = positions[j] - positions[i]
            dist = torch.linalg.norm(disp).item()

            dists.append(dist)
            abs_l1_sum.append(torch.abs(diff).sum().item())
            abs_l2_sum.append((diff**2).sum().item())
            gt_l1_sum.append(torch.abs(gt_block).sum().item())
            gt_l2_sum.append((gt_block**2).sum().item())
            elem_count.append(diff.numel())

            abs_sum = torch.abs(diff).sum().item()
            sq_sum = (diff**2).sum().item()
            gt_abs_sum = torch.abs(gt_block).sum().item()
            gt_sq_sum = (gt_block**2).sum().item()
            n_el = diff.numel()
            edge_l1_abs.append(abs_sum / max(n_el, 1))
            edge_l2_abs.append((sq_sum / max(n_el, 1)) ** 0.5)
            edge_l1_rel.append(abs_sum / max(gt_abs_sum, 1e-14))
            edge_l2_rel.append((sq_sum / max(gt_sq_sum, 1e-14)) ** 0.5)

    if not dists:
        return None

    dists_t = torch.tensor(dists, dtype=torch.float64)
    d_min = float(dists_t.min().item())
    d_max = float(dists_t.max().item())
    if abs(d_max - d_min) < 1e-12:
        d_max = d_min + 1e-6
    bin_edges = torch.linspace(d_min, d_max, n_bins + 1, dtype=torch.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = torch.bucketize(dists_t, bin_edges[1:], right=False).clamp(max=n_bins - 1)

    l1_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    n_edges_per_bin = torch.zeros((n_bins,), dtype=torch.int64)
    n_elems_per_bin = torch.zeros((n_bins,), dtype=torch.int64)

    abs_l1_sum_t = torch.tensor(abs_l1_sum, dtype=torch.float64)
    abs_l2_sum_t = torch.tensor(abs_l2_sum, dtype=torch.float64)
    gt_l1_sum_t = torch.tensor(gt_l1_sum, dtype=torch.float64)
    gt_l2_sum_t = torch.tensor(gt_l2_sum, dtype=torch.float64)
    elem_count_t = torch.tensor(elem_count, dtype=torch.int64)
    # edge_l1_abs_t = torch.tensor(edge_l1_abs, dtype=torch.float64)
    # edge_l2_abs_t = torch.tensor(edge_l2_abs, dtype=torch.float64)
    # edge_l1_rel_t = torch.tensor(edge_l1_rel, dtype=torch.float64)
    # edge_l2_rel_t = torch.tensor(edge_l2_rel, dtype=torch.float64)

    for b in range(n_bins):
        mask = bin_idx == b
        if not torch.any(mask):
            continue
        sum_abs_l1 = abs_l1_sum_t[mask].sum()
        sum_abs_l2 = abs_l2_sum_t[mask].sum()
        sum_gt_l1 = gt_l1_sum_t[mask].sum()
        sum_gt_l2 = gt_l2_sum_t[mask].sum()
        sum_elems = elem_count_t[mask].sum()
        sum_edges = mask.sum()
        n_edges_per_bin[b] = sum_edges
        n_elems_per_bin[b] = sum_elems
        if sum_elems > 0:
            l1_abs[b] = sum_abs_l1 / sum_elems
            l2_abs[b] = torch.sqrt(sum_abs_l2 / sum_elems)
        if sum_gt_l1 > 1e-14:
            l1_rel[b] = sum_abs_l1 / sum_gt_l1
        if sum_gt_l2 > 1e-14:
            l2_rel[b] = torch.sqrt(sum_abs_l2 / sum_gt_l2)

    return {
        "bin_centers": bin_centers.tolist(),
        "l1_abs": l1_abs.tolist(),
        "l2_abs": l2_abs.tolist(),
        "l1_rel": l1_rel.tolist(),
        "l2_rel": l2_rel.tolist(),
        "n_edges": n_edges_per_bin.tolist(),
        "n_elements": n_elems_per_bin.tolist(),
        "d_min": d_min,
        "d_max": d_max,
        "n_bins": n_bins,
    }


def save_distance_error_curve_plot(
    curve_data: dict[str, Any], output_path: Path | str, title: str | None = None
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(curve_data["bin_centers"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("l1_abs", "Abs L1"),
        ("l2_abs", "Abs L2"),
        ("l1_rel", "Rel L1"),
        ("l2_rel", "Rel L2"),
    ]
    for ax, (key, label) in zip(axes.ravel(), plots):
        y = np.asarray(curve_data[key], dtype=float)
        ax.plot(x, y, marker="o", lw=1.4)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    if title:
        fig.suptitle(title)
    _save_plot(fig, output_path)


def save_matrix_comparison_plot(
    pred: BlockMatrix,
    target: BlockMatrix,
    output_path: Path | str,
    *,
    reference: BlockMatrix | None = None,
    title: str = "",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dense = _as_dense(pred)
    target_dense = _as_dense(target)
    diff = pred_dense - target_dense
    if reference is not None:
        ref_dense = _as_dense(reference)
        mu = float(
            torch.sum((pred_dense - target_dense) * ref_dense).item()
            / max(float(torch.sum(ref_dense * ref_dense).item()), 1e-12)
        )
        diff_corr = diff - mu * ref_dense
    else:
        diff_corr = diff.abs()

    vmax = max(
        float(torch.quantile(torch.abs(target_dense.flatten()), 0.8).item()),
        float(torch.quantile(torch.abs(pred_dense.flatten()), 0.8).item()),
        1e-12,
    )
    dmax = max(
        float(torch.quantile(torch.abs(diff.flatten()), 0.8).item()),
        float(torch.quantile(torch.abs(diff_corr.flatten()), 0.8).item()),
        1e-12,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [
        (target_dense, "GT", vmax),
        (pred_dense, "Pred", vmax),
        (diff, "Diff", dmax),
        (diff_corr, "Corr diff" if reference is not None else "Abs diff", dmax),
    ]
    for ax, (mat, label, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(mat.cpu().numpy(), cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
    if title:
        fig.suptitle(title)
    _save_plot(fig, output_path)


def compile_frames_to_gif(
    frame_paths: Iterable[Path], output_path: Path | str, fps: int = 5
) -> None:
    output_path = Path(output_path)
    frames = [Image.open(path).convert("RGB") for path in sorted(frame_paths)]
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    first, rest = frames[0], frames[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def _get_logger_run(trainer: Any):
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return None
    return getattr(logger, "experiment", None)


def _maybe_log_wandb(run: Any, payload: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:
        pass


@dataclass
class ArtifactCheckpointState:
    best_score: float | None = None
    best_epoch: int | None = None


class ArtifactCheckpointCallback(pl.Callback):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        monitor: str = "val/loss_total",
        mode: str = "min",
        save_latest: bool = True,
        save_best: bool = True,
        save_final: bool = True,
        generate_video: bool = False,
        log_per_irrep_images: bool = True,
        distance_bins: int = 16,
        video_fps: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_latest = save_latest
        self.save_best = save_best
        self.save_final = save_final
        self.generate_video = generate_video
        self.log_per_irrep_images = log_per_irrep_images
        self.distance_bins = distance_bins
        self.video_fps = video_fps
        self.state = ArtifactCheckpointState()
        self.reference_batch = None
        self.frames_dir = self.output_dir / "frames"
        self.per_irrep_dir = self.output_dir / "per_irrep_images"
        self.latest_path = self.output_dir / "latest_checkpoint.pt"
        self.best_path = self.output_dir / "best_model.pt"
        self.final_path = self.output_dir / "final_model.pt"

    def _is_better(self, score: float) -> bool:
        if self.state.best_score is None:
            return True
        if self.mode == "min":
            return score < self.state.best_score
        if self.mode == "max":
            return score > self.state.best_score
        raise ValueError(f"Unsupported mode={self.mode!r}")

    def _get_reference_batch(self, trainer: Any):
        loaders = getattr(trainer, "val_dataloaders", None)
        if loaders is None:
            return None
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for loader in loaders:
            try:
                return next(iter(loader))
            except Exception:
                continue
        return None

    def _predict(self, pl_module, batch):
        pl_module.eval()
        with torch.no_grad():
            x, y = batch
            preds = pl_module(x)
        return x, y, preds

    @staticmethod
    def _as_block_matrix(obj, mapper) -> BlockMatrix:
        if isinstance(obj, BlockMatrix):
            return obj
        if isinstance(obj, IrrepsBlockData):
            return obj.to_blocks(mapper)
        raise TypeError(f"Expected BlockMatrix or IrrepsBlockData, got {type(obj)!r}")

    def _maybe_upload_image(self, trainer: Any, key: str, path: Path) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            import wandb

            _maybe_log_wandb(run, {key: wandb.Image(str(path))})
        except Exception:
            return

    def _maybe_upload_video(self, trainer: Any, key: str, path: Path) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            import wandb

            _maybe_log_wandb(
                run, {key: wandb.Video(str(path), fps=self.video_fps, format="gif")}
            )
        except Exception:
            return

    def on_fit_start(self, trainer, pl_module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.per_irrep_dir.mkdir(parents=True, exist_ok=True)
        self.reference_batch = self._get_reference_batch(trainer)
        run = _get_logger_run(trainer)
        if run is not None:
            try:
                run.summary["checkpoint/latest_path"] = str(self.latest_path.resolve())
                run.summary["checkpoint/best_path"] = str(self.best_path.resolve())
                run.summary["checkpoint/final_path"] = str(self.final_path.resolve())
            except Exception:
                pass

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is not None:
            score = float(metric.detach().cpu().item())
            if self.save_best and self._is_better(score):
                trainer.save_checkpoint(str(self.best_path))
                self.state.best_score = score
                self.state.best_epoch = int(trainer.current_epoch)
        if self.save_latest:
            trainer.save_checkpoint(str(self.latest_path))

        if self.reference_batch is None:
            return
        x, y, preds = self._predict(pl_module, self.reference_batch)
        self._save_epoch_frame(trainer, pl_module, x, y, preds)

    def _save_epoch_frame(self, trainer, pl_module, x, y, preds) -> None:
        epoch = int(trainer.current_epoch)
        matrix_names = list(pl_module.cfg.matrix_targets)
        for name in matrix_names:
            if name not in preds or name not in y:
                continue
            pred_mat = self._as_block_matrix(preds[name], pl_module.mapper)
            target_mat = self._as_block_matrix(y[name], pl_module.mapper)
            ref = None
            if name == "hamiltonian" and "overlap" in y:
                ref = self._as_block_matrix(y["overlap"], pl_module.mapper)
            frame_path = self.frames_dir / name / f"epoch_{epoch:04d}.png"
            save_matrix_comparison_plot(
                pred_mat,
                target_mat,
                frame_path,
                reference=ref,
                title=f"{name} epoch {epoch}",
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        if self.save_final:
            trainer.save_checkpoint(str(self.final_path))

        if self.reference_batch is None:
            return
        x, y, preds = self._predict(pl_module, self.reference_batch)

        # final plots
        for name in pl_module.cfg.matrix_targets:
            if name not in preds or name not in y:
                continue
            pred_mat = self._as_block_matrix(preds[name], pl_module.mapper)
            target_mat = self._as_block_matrix(y[name], pl_module.mapper)

            ref = None
            if name == "hamiltonian" and "overlap" in y:
                ref = self._as_block_matrix(y["overlap"], pl_module.mapper)

            if name == "hamiltonian" and "overlap" in y:
                dos_path = self.output_dir / "dos_comparison_final.png"
                save_dos_comparison_plot(pred_mat, target_mat, ref, dos_path)
                self._maybe_upload_image(trainer, "final/dos_comparison_plot", dos_path)

            curve = compute_distance_error_curve(
                pred_mat,
                target_mat,
                x["positions"],
                x.get("box"),
                n_bins=self.distance_bins,
            )
            if curve is not None:
                curve_path = self.output_dir / f"distance_error_curve_{name}.png"
                save_distance_error_curve_plot(
                    curve, curve_path, title=f"{name} distance error"
                )
                self._maybe_upload_image(
                    trainer, f"distance_curve/{name}_plot", curve_path
                )

            if self.log_per_irrep_images:
                per_irrep_subdir = self.per_irrep_dir / name
                per_irrep_subdir.mkdir(parents=True, exist_ok=True)
                target_irreps = target_mat.to_vectors(pl_module.mapper)
                pred_irreps = pred_mat.to_vectors(pl_module.mapper)
                all_irreps = get_all_irreps(pl_module.mapper)
                for irrep in all_irreps:
                    pred_ir = filter_irreps_block_data_by_irrep(
                        pred_irreps, irrep, pl_module.mapper
                    ).to_blocks(pl_module.mapper)
                    tgt_ir = filter_irreps_block_data_by_irrep(
                        target_irreps, irrep, pl_module.mapper
                    ).to_blocks(pl_module.mapper)
                    if not pred_ir.pair_blocks or not tgt_ir.pair_blocks:
                        continue
                    img_path = per_irrep_subdir / f"{irrep}.png"
                    save_matrix_comparison_plot(
                        pred_ir,
                        tgt_ir,
                        img_path,
                        reference=ref,
                        title=f"{name} / {irrep}",
                    )
                    self._maybe_upload_image(
                        trainer, f"irrep_images/{name}/{irrep}", img_path
                    )

        if self.generate_video:
            for name in pl_module.cfg.matrix_targets:
                frame_dir = self.frames_dir / name
                if not frame_dir.exists():
                    continue
                video_path = self.output_dir / f"training_progress_{name}.gif"
                compile_frames_to_gif(
                    frame_dir.glob("*.png"), video_path, fps=self.video_fps
                )
                if video_path.exists():
                    self._maybe_upload_video(
                        trainer, f"training_video_{name}", video_path
                    )
