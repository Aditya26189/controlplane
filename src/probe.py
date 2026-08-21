"""The probe: standardisation, logistic regression, layer sweep, threshold.

Features are the residual-stream vector at one layer, taken at the final prompt
token. The label is ``1`` when the generated answer was **incorrect** -- the
positive class is the thing we want to catch (DECISIONS.md 004). Getting that
backwards produces ``AUROC = 1 - true_auroc``, which reads as a strong negative
result rather than as a bug, so the polarity is asserted rather than assumed.

Selection discipline (CLAUDE.md invariant 2): the layer, the regularisation
strength and the threshold are all chosen on **validation**. Nothing in this
module reads test rows -- ``run_sweep`` is handed the split labels and filters
to train and val only. Test is scored once, by ``scripts/02_train_probe.py``.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import Config

LOGGER = logging.getLogger(__name__)

POSITIVE_CLASS_MEANING = "incorrect"


class PolarityError(ValueError):
    """Raised when the label vector does not carry the documented polarity."""


def assert_polarity(labels: np.ndarray, correct: Optional[np.ndarray] = None) -> None:
    """Assert ``label == 1`` means "the generated answer was incorrect".

    Enforces DECISIONS.md 004 at the boundary between extraction and the probe.
    Both classes must be present as well: a single-class label vector cannot
    train a classifier, and sklearn's error for that is opaque.

    Args:
        labels: Integer labels, one per example.
        correct: Optional boolean correctness column; if given, must be the
            exact complement of ``labels``.

    Raises:
        PolarityError: on non-binary labels, a missing class, or a mismatch
            against ``correct``.
    """
    unique = set(np.unique(labels).tolist())
    if not unique <= {0, 1}:
        raise PolarityError(f"labels must be 0/1, found {sorted(unique)}")
    if unique != {0, 1}:
        raise PolarityError(
            f"labels contain only class {unique}; a probe needs both. Either the "
            "model got everything right, everything wrong, or the matching rule "
            "is broken."
        )
    if correct is not None:
        expected = (~np.asarray(correct).astype(bool)).astype(int)
        if not np.array_equal(np.asarray(labels).astype(int), expected):
            raise PolarityError(
                "label is not the complement of correct. The positive class must "
                f"be {POSITIVE_CLASS_MEANING!r} (DECISIONS.md 004); inverting it "
                "silently yields 1 - AUROC."
            )


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #


def build_scaler(config: Config) -> StandardScaler:
    """Create the feature scaler, honouring ``probe.standardize``.

    Residual-stream vectors have large, layer-varying magnitudes, so an
    unstandardised logistic regression is badly conditioned (CLAUDE.md pitfall
    list). When standardisation is off, a no-op ``StandardScaler`` is still
    returned so the transform path is identical either way.
    """
    if config.probe.standardize:
        return StandardScaler()
    return StandardScaler(with_mean=False, with_std=False)


def fit_probe(
    x_train: np.ndarray, y_train: np.ndarray, c_value: float, config: Config
) -> tuple[StandardScaler, LogisticRegression]:
    """Fit the scaler and the logistic regression on training rows only.

    The scaler is fitted here and nowhere else. Fitting it on the full set --
    or refitting it on val or test -- leaks distributional information across
    the split boundary (CLAUDE.md pitfall list), and it is a one-line mistake.

    ``class_weight='balanced'`` because the label distribution is unbalanced: if
    the model is right 65% of the time, an unweighted fit is rewarded for
    predicting "correct" every time.

    Args:
        x_train: ``(n_train, hidden)`` activations.
        y_train: ``(n_train,)`` labels, 1 == incorrect.
        c_value: Inverse regularisation strength.
        config: Resolved experiment config.

    Returns:
        ``(fitted scaler, fitted classifier)``.
    """
    assert_polarity(y_train)
    scaler = build_scaler(config).fit(x_train)
    classifier = LogisticRegression(
        max_iter=config.probe.max_iter,
        class_weight=config.probe.class_weight,
        C=c_value,
        random_state=config.seed,
    )
    classifier.fit(scaler.transform(x_train), y_train)
    return scaler, classifier


def probe_scores(
    scaler: StandardScaler, classifier: LogisticRegression, x: np.ndarray
) -> np.ndarray:
    """Score examples: higher means "more likely to be wrong".

    Uses ``decision_function`` rather than ``predict_proba`` because it is the
    raw dot product -- the quantity whose cost is measured in ``latency.py`` --
    and because it is monotone in the probability, so AUROC and any
    quantile-based threshold are unchanged.
    """
    return classifier.decision_function(scaler.transform(x))


# --------------------------------------------------------------------------- #
# Layer x C sweep, on validation only
# --------------------------------------------------------------------------- #


def _split_masks(split: Sequence[str]) -> dict[str, np.ndarray]:
    """Boolean masks per split name."""
    arr = np.asarray(split)
    return {name: arr == name for name in ("train", "val", "test")}


def run_sweep(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    split: Sequence[str],
    config: Config,
) -> dict[str, Any]:
    """Sweep layers x C, training on train and scoring AUROC on validation.

    Test rows are filtered out at the top of this function and never referenced
    again, which is what makes CLAUDE.md invariant 2 checkable by reading rather
    than by trusting. The full table is returned, not just the winner: a smooth
    curve peaking mid-stack is itself evidence the signal is real, and a
    reviewer wants to see its shape.

    Args:
        activations: Layer index -> ``(n, hidden)`` array, all splits.
        labels: ``(n,)`` labels, 1 == incorrect.
        split: ``(n,)`` split names.
        config: Resolved experiment config.

    Returns:
        ``{"sweep": [...], "best": {...}}`` with one entry per (layer, C).
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    assert_polarity(labels)
    masks = _split_masks(split)
    train_mask, val_mask = masks["train"], masks["val"]
    if not train_mask.any() or not val_mask.any():
        raise ValueError("sweep needs a non-empty train and validation split")

    y_train, y_val = labels[train_mask], labels[val_mask]
    assert_polarity(y_train)
    assert_polarity(y_val)

    rows: list[dict[str, Any]] = []
    for layer in sorted(activations):
        x = activations[layer]
        if x.shape[0] != labels.shape[0]:
            raise AssertionError(
                f"layer {layer}: {x.shape[0]} activations for {labels.shape[0]} labels"
            )
        x_train, x_val = x[train_mask], x[val_mask]
        for c_value in config.probe.C_grid:
            scaler, classifier = fit_probe(x_train, y_train, c_value, config)
            val_auroc = float(roc_auc_score(y_val, probe_scores(scaler, classifier, x_val)))
            train_auroc = float(
                roc_auc_score(y_train, probe_scores(scaler, classifier, x_train))
            )
            rows.append(
                {
                    "layer": int(layer),
                    "C": float(c_value),
                    "val_auroc": val_auroc,
                    "train_auroc": train_auroc,
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                }
            )
            LOGGER.info(
                "layer %2d  C=%-6g  val AUROC %.4f  (train %.4f)",
                layer,
                c_value,
                val_auroc,
                train_auroc,
            )

    best = select_best(rows)
    LOGGER.info(
        "selected layer %d with C=%g on validation (AUROC %.4f)",
        best["layer"],
        best["C"],
        best["val_auroc"],
    )
    return {"sweep": rows, "best": best}


def select_best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest validation AUROC, breaking ties deterministically.

    Ties go to the shallower layer and then to the stronger regularisation
    (smaller C). Both are the conservative choice, and fixing the rule means two
    runs of the same sweep cannot disagree about the winner.
    """
    if not rows:
        raise ValueError("cannot select a winner from an empty sweep")
    return dict(min(rows, key=lambda r: (-r["val_auroc"], r["layer"], r["C"])))


# --------------------------------------------------------------------------- #
# Threshold selection, on validation only
# --------------------------------------------------------------------------- #


def select_threshold(scores_val: np.ndarray, target_flag_rate: float) -> float:
    """Choose the score threshold that hits the target flag rate on validation.

    Frozen here and applied unchanged to test. The test flag rate will differ
    from the target -- every downstream calculation therefore uses the
    *measured* test rate, never this target (CLAUDE.md invariant 6).

    Args:
        scores_val: Validation probe scores.
        target_flag_rate: Desired fraction of responses flagged.

    Returns:
        The threshold; an example is flagged when ``score >= threshold``.
    """
    scores = np.asarray(scores_val, dtype=float)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("cannot choose a threshold from an empty validation set")
    k = max(1, int(round(target_flag_rate * n)))
    k = min(k, n)
    threshold = float(np.sort(scores)[::-1][k - 1])
    achieved = float(np.mean(scores >= threshold))
    LOGGER.info(
        "threshold %.6f chosen on validation: target flag rate %.3f, achieved %.3f "
        "(%d of %d)",
        threshold,
        target_flag_rate,
        achieved,
        int(np.sum(scores >= threshold)),
        n,
    )
    return threshold


# --------------------------------------------------------------------------- #
# The fitted probe
# --------------------------------------------------------------------------- #


@dataclass
class FittedProbe:
    """A trained probe with everything needed to score new activations.

    Carries its own provenance -- layer, C, threshold, and the validation
    numbers it was selected on -- so a saved probe cannot be separated from the
    decisions that produced it.
    """

    layer: int
    c_value: float
    scaler: StandardScaler
    classifier: LogisticRegression
    threshold: float
    target_flag_rate: float
    val_auroc: float
    val_flag_rate: float
    n_train: int
    hidden_size: int
    positive_class: str = POSITIVE_CLASS_MEANING

    def score(self, x: np.ndarray) -> np.ndarray:
        """Score activations; higher means more likely to be incorrect."""
        return probe_scores(self.scaler, self.classifier, x)

    def flags(self, x: np.ndarray) -> np.ndarray:
        """Boolean flag decisions at the frozen threshold."""
        return self.score(x) >= self.threshold

    def to_meta(self) -> dict[str, Any]:
        """JSON-serialisable summary for the results artifacts."""
        return {
            "layer": self.layer,
            "C": self.c_value,
            "threshold": self.threshold,
            "target_flag_rate": self.target_flag_rate,
            "val_auroc": self.val_auroc,
            "val_flag_rate": self.val_flag_rate,
            "n_train": self.n_train,
            "hidden_size": self.hidden_size,
            "positive_class": self.positive_class,
            "standardized": bool(getattr(self.scaler, "with_mean", True)),
        }


def fit_selected_probe(
    activations: dict[int, np.ndarray],
    labels: np.ndarray,
    split: Sequence[str],
    best: dict[str, Any],
    config: Config,
) -> FittedProbe:
    """Refit the winning (layer, C) on train and freeze its threshold on val.

    Args:
        activations: Layer index -> ``(n, hidden)`` array.
        labels: ``(n,)`` labels, 1 == incorrect.
        split: ``(n,)`` split names.
        best: The winning row from :func:`run_sweep`.
        config: Resolved experiment config.

    Returns:
        The fitted probe, threshold already frozen.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    masks = _split_masks(split)
    layer = int(best["layer"])
    x = activations[layer]

    x_train, y_train = x[masks["train"]], labels[masks["train"]]
    x_val, y_val = x[masks["val"]], labels[masks["val"]]

    scaler, classifier = fit_probe(x_train, y_train, float(best["C"]), config)
    scores_val = probe_scores(scaler, classifier, x_val)
    threshold = select_threshold(scores_val, config.economics.target_flag_rate)

    return FittedProbe(
        layer=layer,
        c_value=float(best["C"]),
        scaler=scaler,
        classifier=classifier,
        threshold=threshold,
        target_flag_rate=config.economics.target_flag_rate,
        val_auroc=float(roc_auc_score(y_val, scores_val)),
        val_flag_rate=float(np.mean(scores_val >= threshold)),
        n_train=int(masks["train"].sum()),
        hidden_size=int(x.shape[1]),
    )


def save_probe(probe: FittedProbe, path: str | Path) -> Path:
    """Persist the fitted probe with joblib."""
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, out)
    LOGGER.info("wrote %s", out)
    return out


def load_probe(path: str | Path) -> FittedProbe:
    """Load a fitted probe written by :func:`save_probe`."""
    import joblib

    probe = joblib.load(path)
    if not isinstance(probe, FittedProbe):
        raise TypeError(f"{path} does not contain a FittedProbe")
    return probe
