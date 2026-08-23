"""Plots. Every one carries the provenance of the numbers in it.

Two rules, both enforced here rather than remembered:

* **An estimated value is never drawn without its interval.** Invariant 4 is a
  rendering rule as much as a data rule, and a bar chart of point estimates is
  the most persuasive way to publish an unbacked claim.
* **A synthetic run is stamped as synthetic, on the figure.** A plot outlives
  the terminal it was produced in, and the caption is what someone reads six
  weeks later. ``DECISIONS.md`` 027.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # no display on a Kaggle worker or in CI
import matplotlib.pyplot as plt  # noqa: E402

from ..model import MetricKind  # noqa: E402
from ..validation.ablation import TierLadder  # noqa: E402
from ..validation.evalsets import SOURCE_SYNTHETIC  # noqa: E402

__all__ = ["plot_tier_ladder"]

_LOG = logging.getLogger(__name__)

_STATUS_COLOUR = {
    "VALID": "#2a7d4f",
    "REFUSED": "#b3261e",
    "STALE": "#b06f00",
    "REVOKED": "#7a2f8a",
}


def plot_tier_ladder(
    ladder: TierLadder,
    path: str | Path,
    *,
    metrics: tuple[str, ...] = ("auroc", "recall", "precision"),
    config_hash: Optional[str] = None,
    git_commit: Optional[str] = None,
) -> Path:
    """Draw the tier ladder with intervals, one panel per metric.

    Precision and recall appear together by default (invariant 5). AUROC is
    shown beside them with the base rate in the caption, because AUROC without a
    base rate is how an imbalanced dataset flatters a constant predictor.

    Args:
        ladder: The measured ladder.
        path: Output PNG path.
        metrics: Which metrics to panel. Each must be estimated — an exact count
            has no interval and does not belong on an interval plot.
        config_hash: Stamped into the caption.
        git_commit: Stamped into the caption.

    Returns:
        The path written.

    Raises:
        ValueError: If asked to plot an exact metric, which would be drawn as a
            bare point and read as an estimate.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    variants = [run.variant for run in ladder.runs]
    y = range(len(variants))

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.2 * len(metrics), 0.6 * len(variants) + 3.2),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        for row, run in enumerate(ladder.runs):
            metric = ladder.metric(run.variant, metric_name)
            if metric.kind is not MetricKind.ESTIMATED:
                raise ValueError(
                    f"{metric_name} is {metric.kind.value}; an exact count has no "
                    "interval and would be drawn as a bare point, which reads as "
                    "an estimate (CLAUDE.md invariant 4)"
                )
            colour = _STATUS_COLOUR.get(run.warrant.status.value, "#555555")
            ax.errorbar(
                metric.value,
                row,
                xerr=[[metric.value - metric.ci_low], [metric.ci_high - metric.value]],
                fmt="o",
                color=colour,
                ecolor=colour,
                elinewidth=2,
                capsize=4,
                markersize=7,
            )
        ax.set_title(f"{metric_name}  (95% CI)")
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        if metric_name == "auroc":
            ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1)
            ax.annotate(
                "chance", xy=(0.5, -0.7), fontsize=8, color="#777777", ha="center"
            )

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(
        [
            f"{run.variant}\n{run.warrant.access_tier.name} · {run.warrant.status.value}"
            for run in ladder.runs
        ],
        fontsize=8,
    )
    axes[0].invert_yaxis()

    synthetic = ladder.data_source == SOURCE_SYNTHETIC
    title = f"Tier ladder — {ladder.eval_set_id}"
    if synthetic:
        title = f"[SYNTHETIC FIXTURE — NOT A MEASUREMENT]  {title}"
    fig.suptitle(title, fontsize=12, color="#b3261e" if synthetic else "#111111")

    caption_parts = [
        f"envelope {ladder.envelope_id}",
        f"base rate {ladder.base_rate:.4f}",
        f"n_test {ladder.runs[0].warrant.n_test}" if ladder.runs else "",
        f"data_source {ladder.data_source}",
    ]
    if config_hash:
        caption_parts.append(f"config {config_hash}")
    if git_commit:
        caption_parts.append(f"commit {git_commit[:8]}")
    fig.text(
        0.5,
        0.015,
        " · ".join(p for p in caption_parts if p),
        ha="center",
        fontsize=8,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _LOG.info("wrote %s", out)
    return out
