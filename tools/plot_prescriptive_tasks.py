#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dataset(path: Path) -> dict:
    if path.is_dir():
        path = path / "dataset.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset.pt at {path}")
    return torch.load(path, weights_only=False)


def _load_edge_collect_module():
    module_path = _repo_root() / "tasks" / "edge_forbid" / "collect" / "collect_dataset.py"
    spec = importlib.util.spec_from_file_location("edge_collect", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _coords_to_matrix(coords: np.ndarray) -> np.ndarray:
    mod = _load_edge_collect_module()
    return mod.coords_to_ceil2d_int_matrix(torch.from_numpy(coords))


def _tour_length(D: np.ndarray, tour: Iterable[int]) -> float:
    tour_idx = np.asarray(list(tour), dtype=int)
    nxt = np.roll(tour_idx, -1)
    return float(D[tour_idx, nxt].sum())


def _solve_tour(D: np.ndarray, *, forbid_edge: Tuple[int, int] | None = None) -> list[int]:
    mod = _load_edge_collect_module()
    D_use = D.copy()
    if forbid_edge is not None:
        i, j = forbid_edge
        forbid_cost = int(D_use.max()) + 10_000_000
        D_use[i, j] = forbid_cost
        D_use[j, i] = forbid_cost

    scratch_dir = Path(tempfile.mkdtemp(prefix="plot_tasks_"))
    try:
        res = mod.solve_with_concorde_matrix(D_use, scratch_dir=scratch_dir, base_stub="inst", timeout_sec=60.0)
    finally:
        for p in scratch_dir.iterdir():
            p.unlink()
        scratch_dir.rmdir()

    if not res.valid or res.tour0 is None:
        raise RuntimeError("Concorde failed to return a valid tour.")
    return res.tour0


def _plot_tour(ax, coords: np.ndarray, tour: Iterable[int], *, color: str, lw: float, alpha: float) -> None:
    tour_idx = list(tour)
    pts = coords[np.asarray(tour_idx, dtype=int)]
    pts = np.vstack([pts, pts[0]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha)


def _plot_tour_masked(
    ax,
    coords: np.ndarray,
    tour: Iterable[int],
    mask_fn,
    *,
    color: str,
    lw: float,
    alpha: float,
    linestyle: str = "solid",
    zorder: int | None = None,
) -> None:
    tour_idx = np.asarray(list(tour), dtype=int)
    pts = coords[tour_idx]
    pts_next = np.roll(pts, -1, axis=0)
    mids = (pts + pts_next) * 0.5
    keep = mask_fn(mids)
    if not np.any(keep):
        return
    segs = np.stack([pts[keep], pts_next[keep]], axis=1)
    lc = __import__("matplotlib.collections").collections.LineCollection(
        segs, colors=color, linewidths=lw, alpha=alpha, linestyles=linestyle
    )
    if zorder is not None:
        lc.set_zorder(zorder)
    ax.add_collection(lc)


def _node_removal_figure(
    coords: np.ndarray,
    delta: np.ndarray,
    best_idx: int,
    base_tour: list[int],
    removed_tour: list[int],
    removed_idx: int,
    base_length: float,
    removed_length: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=160)
    ax = axes[0]

    _plot_tour(ax, coords, base_tour, color="black", lw=2.5, alpha=0.5)
    finite_delta = delta[np.isfinite(delta)]
    vmax = float(np.nanmax(finite_delta)) if finite_delta.size else None
    if vmax and vmax > 0:
        positive = finite_delta[finite_delta > 0]
        vmin = float(np.nanmin(positive)) if positive.size else vmax * 0.02
        vmin = max(vmin, vmax * 0.02)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        delta_plot = np.where(np.isfinite(delta) & (delta > 0), delta, vmin)
    else:
        norm = None
        delta_plot = delta
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=delta_plot,
        s=94,
        cmap="viridis",
        norm=norm,
        edgecolors="black",
        linewidths=0.4,
        zorder=8,
    )
    best_xy = coords[best_idx]
    ax.add_patch(
        patches.Circle(
            (best_xy[0], best_xy[1]),
            radius=0.0375,
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
            alpha=0.9,
            zorder=10,
        )
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="4%", pad=0.15)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"$\Delta$ length if node is removed", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    if norm is not None:
        ticks = np.linspace(norm.vmin, norm.vmax, 6)
        labels = []
        for t in ticks:
            scaled = -5.0 * (t - norm.vmin) / (norm.vmax - norm.vmin)
            labels.append("0%" if abs(scaled) < 0.05 else f"{scaled:.0f}%")
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    ax2 = axes[1]
    _plot_tour(ax2, coords, removed_tour, color="tab:red", lw=2.5, alpha=0.65)
    ax2.scatter(coords[:, 0], coords[:, 1], c="black", s=56, edgecolors="none", alpha=0.7, zorder=8)
    ax2.add_patch(
        patches.Circle(
            (best_xy[0], best_xy[1]),
            radius=0.0375,
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
            alpha=0.9,
            zorder=10,
        )
    )
    ax2.set_axis_off()
    ax2.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.06, wspace=0.08)
    pos_left = ax.get_position()
    pos_right = ax2.get_position()
    title_y = min(max(pos_left.y1, pos_right.y1) + 0.035, 0.97)
    subtitle_y = title_y - 0.055
    fig.text(
        (pos_left.x0 + pos_left.x1) * 0.5,
        title_y,
        "Optimal tour (base)",
        ha="center",
        va="bottom",
        fontsize=22,
    )
    fig.text(
        (pos_left.x0 + pos_left.x1) * 0.5,
        subtitle_y,
        f"Length: {base_length:.0f}",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    fig.text(
        (pos_right.x0 + pos_right.x1) * 0.5,
        title_y,
        "Optimal tour (top node removed)",
        ha="center",
        va="bottom",
        fontsize=22,
    )
    fig.text(
        (pos_right.x0 + pos_right.x1) * 0.5,
        subtitle_y,
        f"Length: {removed_length:.0f}",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _edge_forbid_figure(
    coords: np.ndarray,
    tour: np.ndarray,
    delta: np.ndarray,
    best_edge_idx: int,
    forbid_tour: list[int],
    base_length: float,
    forbid_length: float,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors
    from matplotlib import patches
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), dpi=160)
    ax = axes[0]
    pts = coords[tour]
    segs = np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)
    finite_delta = delta[np.isfinite(delta)]
    vmax = float(np.percentile(finite_delta, 99.5)) if finite_delta.size else None
    if vmax and vmax > 0:
        positive = finite_delta[finite_delta > 0]
        vmin = float(np.nanmin(positive)) if positive.size else vmax * 0.02
        vmin = max(vmin, vmax * 0.02)
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        delta_plot = np.where(delta > 0, delta, vmin)
    else:
        norm = None
        delta_plot = delta
    lc_outline = LineCollection(segs, colors="black", linewidths=6.5, alpha=1.0)
    ax.add_collection(lc_outline)
    lc = LineCollection(segs, array=delta_plot, cmap="viridis", linewidths=5.0, alpha=1.0, norm=norm)
    ax.add_collection(lc)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c="black",
        s=40,
        edgecolors="none",
        alpha=0.7,
        zorder=8,
    )
    u = int(tour[best_edge_idx])
    v = int(tour[(best_edge_idx + 1) % tour.size])
    mid = 0.5 * (coords[u] + coords[v])
    edge_len = float(np.linalg.norm(coords[u] - coords[v]))
    angle = float(np.degrees(np.arctan2(coords[v, 1] - coords[u, 1], coords[v, 0] - coords[u, 0])))
    ellipse = patches.Ellipse(
        (mid[0], mid[1]),
        width=edge_len * 5.2,
        height=max(0.08, edge_len * 0.5),
        angle=angle,
        fill=False,
        edgecolor="tab:red",
        linewidth=2.0,
        alpha=0.9,
        zorder=9,
    )
    ax.add_patch(ellipse)
    ax2 = axes[1]
    _plot_tour(ax2, coords, forbid_tour, color="tab:red", lw=2.5, alpha=0.65)
    ax2.scatter(coords[:, 0], coords[:, 1], c="black", s=40, edgecolors="none", alpha=0.7, zorder=8)
    ax2.add_patch(
        patches.Ellipse(
            (mid[0], mid[1]),
            width=edge_len * 5.2,
            height=max(0.08, edge_len * 0.5),
            angle=angle,
            fill=False,
            edgecolor="tab:red",
            linewidth=2.0,
            alpha=0.9,
            zorder=9,
        )
    )
    ax2.set_axis_off()
    ax2.set_aspect("equal", adjustable="box")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="4%", pad=0.15)
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label(r"$\Delta$ length if edge is forbidden", fontsize=18)
    cb.ax.tick_params(labelsize=15)
    if norm is not None:
        ticks = np.linspace(norm.vmin, norm.vmax, 6)
        labels = []
        for t in ticks:
            scaled = 5.0 * (t - norm.vmin) / (norm.vmax - norm.vmin)
            labels.append("0%" if abs(scaled) < 0.05 else f"{scaled:.0f}%")
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    cax.yaxis.set_label_position("left")
    cax.yaxis.tick_left()
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.06, wspace=0.08)
    pos_left = ax.get_position()
    pos_right = ax2.get_position()
    title_y = min(max(pos_left.y1, pos_right.y1) + 0.035, 0.97)
    subtitle_y = title_y - 0.055
    fig.text(
        (pos_left.x0 + pos_left.x1) * 0.5,
        title_y,
        "Optimal tour (base)",
        ha="center",
        va="bottom",
        fontsize=22,
    )
    fig.text(
        (pos_left.x0 + pos_left.x1) * 0.5,
        subtitle_y,
        f"Length: {base_length:.0f}",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    fig.text(
        (pos_right.x0 + pos_right.x1) * 0.5,
        title_y,
        "Optimal tour (top edge forbidden)",
        ha="center",
        va="bottom",
        fontsize=22,
    )
    fig.text(
        (pos_right.x0 + pos_right.x1) * 0.5,
        subtitle_y,
        f"Length: {forbid_length:.0f}",
        ha="center",
        va="bottom",
        fontsize=18,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render prescriptive task figures for the paper.")
    p.add_argument("--node_data_dir", type=str, required=True)
    p.add_argument("--node_instance", type=int, default=0)
    p.add_argument("--edge_data_dir", type=str, required=True)
    p.add_argument("--edge_instance", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="papers/icml/imgs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()

    node_ds = _load_dataset(Path(args.node_data_dir).expanduser().resolve())
    edge_ds = _load_dataset(Path(args.edge_data_dir).expanduser().resolve())

    # Node-removal task.
    coords = node_ds["locs"][args.node_instance].detach().cpu().numpy().astype(np.float64, copy=False)
    delta = node_ds["delta_length_pct"][args.node_instance].detach().cpu().numpy().astype(np.float64, copy=False)
    valid = node_ds["valid_minus"][args.node_instance].detach().cpu().numpy().astype(bool, copy=False)
    delta = np.where(valid, delta, np.nan)
    best_idx = int(np.nanargmax(delta))

    D = _coords_to_matrix(coords)
    base_tour = _solve_tour(D)

    keep = [i for i in range(coords.shape[0]) if i != best_idx]
    coords_minus = coords[keep]
    D_minus = D[np.ix_(keep, keep)]
    tour_minus = _solve_tour(D_minus)
    # Map reduced tour indices back to original indices.
    mapped_minus = [keep[i] for i in tour_minus]

    base_length = _tour_length(D, base_tour)
    removed_length = _tour_length(D_minus, tour_minus)
    _node_removal_figure(
        coords=coords,
        delta=delta,
        best_idx=best_idx,
        base_tour=base_tour,
        removed_tour=mapped_minus,
        removed_idx=best_idx,
        base_length=base_length,
        removed_length=removed_length,
        out_path=out_dir / "task_node_removal.png",
    )

    # Edge-forbid task.
    coords = edge_ds["locs"][args.edge_instance].detach().cpu().numpy().astype(np.float64, copy=False)
    tour = edge_ds["base_tour"][args.edge_instance].detach().cpu().numpy().astype(np.int64, copy=False)
    delta = edge_ds["delta_length_pct"][args.edge_instance].detach().cpu().numpy().astype(np.float64, copy=False)
    valid = edge_ds["valid_forbid"][args.edge_instance].detach().cpu().numpy().astype(bool, copy=False)
    delta = np.where(valid, delta, np.nan)
    best_edge_idx = int(np.nanargmax(delta))

    D = _coords_to_matrix(coords)
    u = int(tour[best_edge_idx])
    v = int(tour[(best_edge_idx + 1) % tour.size])
    forbid_tour = _solve_tour(D, forbid_edge=(u, v))

    base_length = _tour_length(D, tour)
    forbid_length = _tour_length(D, forbid_tour)
    _edge_forbid_figure(
        coords=coords,
        tour=tour,
        delta=delta,
        best_edge_idx=best_edge_idx,
        forbid_tour=forbid_tour,
        base_length=base_length,
        forbid_length=forbid_length,
        out_path=out_dir / "task_edge_forbid.png",
    )

    print(f"[plots] wrote: {out_dir / 'task_node_removal.png'}")
    print(f"[plots] wrote: {out_dir / 'task_edge_forbid.png'}")


if __name__ == "__main__":
    main()
