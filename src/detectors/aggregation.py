"""Pooling a sequence of residual-stream vectors into one feature vector.

Two strategies, and the difference between them is the whole point of the
long-context beat.

**Mean pooling** averages every position. On a 200-token prompt that is a
reasonable summary. On a 16,000-token prompt where the answer-relevant span is
40 tokens long, the signal is one part in four hundred and the average is
dominated by the distractors. This is the documented failure mode
(``CLAUDE.md``, "Silent failures"): mean-pooled linear probes collapse under
long-context shift, and they collapse *quietly* — the probe still returns a
score, and the score is near chance.

**Max of rolling means** takes overlapping windows, averages within each, then
takes the element-wise maximum across windows. A local signal survives because
it dominates its own window even when it is a rounding error in the global
average. The cost is a noisier feature: taking a maximum over many windows
selects for extremes, so the estimator has more variance and the resulting
warrant carries a wider interval. That trade is the finding, not a defect —
a wider interval that is *true* on long context beats a narrow one that is not.

Both are computed at extraction and both are warranted separately
(``SPEC.md`` §3.1), so the matrix can refuse one while holding the other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["AggregationError", "aggregate", "mean_pool", "max_rolling_means"]


class AggregationError(ValueError):
    """Raised when a sequence cannot be pooled as asked."""


def _validate(hidden: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Check shapes and return a boolean mask over real (non-pad) positions.

    The mask is not optional in spirit. Pooling across pad positions is the
    same class of error as reading position −1 of a right-padded batch: it
    produces a plausible vector built partly from nothing.
    """
    if hidden.ndim != 2:
        raise AggregationError(
            f"expected (seq_len, hidden_dim), got shape {hidden.shape}"
        )
    seq_len = hidden.shape[0]
    if mask is None:
        return np.ones(seq_len, dtype=bool)
    mask = np.asarray(mask).astype(bool)
    if mask.shape != (seq_len,):
        raise AggregationError(
            f"mask shape {mask.shape} does not match sequence length {seq_len}"
        )
    if not mask.any():
        raise AggregationError(
            "attention mask selects no positions; pooling would average over "
            "padding alone"
        )
    return mask


def mean_pool(hidden: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean over real positions.

    Args:
        hidden: ``(seq_len, hidden_dim)`` residual-stream activations.
        mask: ``(seq_len,)`` attention mask; pad positions excluded.

    Returns:
        ``(hidden_dim,)``.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    real = _validate(hidden, mask)
    return hidden[real].mean(axis=0)


def max_rolling_means(
    hidden: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    window: int,
    stride: int,
) -> np.ndarray:
    """Element-wise maximum over the means of overlapping windows.

    Windows overlap by design. With ``stride == window`` a signal straddling a
    boundary is split across two windows and diluted in both, which reintroduces
    the failure the strategy exists to avoid. A stride of half the window means
    every span of ``window`` tokens sits wholly inside at least one window.

    Sequences shorter than one window are pooled as a single window, so this
    degenerates to :func:`mean_pool` on short inputs — which is why the two
    strategies agree on ``triviaqa-600`` and diverge on ``triviaqa-longctx-600``.

    Args:
        hidden: ``(seq_len, hidden_dim)`` residual-stream activations.
        mask: ``(seq_len,)`` attention mask; pad positions excluded before
            windowing, so windows are over real tokens.
        window: Window length in tokens.
        stride: Step between window starts.

    Returns:
        ``(hidden_dim,)``.

    Raises:
        AggregationError: If the window or stride is not positive, or the stride
            exceeds the window and would leave gaps.
    """
    if window <= 0 or stride <= 0:
        raise AggregationError(
            f"window and stride must be positive, got window={window}, stride={stride}"
        )
    if stride > window:
        raise AggregationError(
            f"stride {stride} exceeds window {window}, which leaves positions in "
            "no window at all. A signal landing in a gap is invisible to the probe."
        )
    hidden = np.asarray(hidden, dtype=np.float64)
    real = _validate(hidden, mask)
    kept = hidden[real]
    n = kept.shape[0]
    if n <= window:
        return kept.mean(axis=0)

    starts = range(0, n - window + 1, stride)
    means = [kept[s : s + window].mean(axis=0) for s in starts]
    # A trailing partial window is included when the stride does not divide the
    # sequence: dropping it would make the last tokens of every long prompt
    # invisible, and the end of a prompt is where the question usually is.
    last_start = (n - window) - ((n - window) % stride)
    if last_start + window < n:
        means.append(kept[n - window :].mean(axis=0))
    return np.max(np.stack(means, axis=0), axis=0)


def aggregate(
    hidden: np.ndarray,
    strategy: str,
    mask: np.ndarray | None = None,
    *,
    window: int,
    stride: int,
) -> np.ndarray:
    """Dispatch to a named pooling strategy.

    Args:
        hidden: ``(seq_len, hidden_dim)``.
        strategy: One of ``config.probe.aggregations``.
        mask: Attention mask.
        window: Window length for ``max_rolling_means``.
        stride: Window stride for ``max_rolling_means``.

    Returns:
        ``(hidden_dim,)``.

    Raises:
        AggregationError: On an unknown strategy. Not a silent fallback to mean
            pooling: a typo would then produce the collapsing strategy under the
            robust one's name, and the matrix would warrant the wrong thing.
    """
    if strategy == "mean_pool":
        return mean_pool(hidden, mask)
    if strategy == "max_rolling_means":
        return max_rolling_means(hidden, mask, window=window, stride=stride)
    if strategy == "last_token":
        # The Round 1 strategy: the final real position of the prompt, before
        # any token has been generated. Kept because it is the anchor the Round 1
        # number was measured with.
        real = _validate(hidden, mask)
        return np.asarray(hidden, dtype=np.float64)[real][-1]
    raise AggregationError(
        f"unknown aggregation {strategy!r}. Not falling back to mean pooling: a "
        "typo would then run the collapsing strategy under the robust one's name."
    )
