"""Wall-clock cost of the probe against the generation it rides along with.

The competition brief asks directly how the system avoids slowing the model
down. A measured ratio is a far stronger answer than a claim, so nothing here
is asserted: the probe is timed on a real activation vector, and generation
time comes from the ``generate()`` calls recorded during extraction (SPEC.md
§8).

The honest framing, which ``report.py`` repeats: the probe adds **no additional
forward pass**. The activation is a by-product of the prefill that generation
already performs, so the marginal cost of the probe is the scale-and-dot-product
timed here, and nothing else.
"""

import logging
import time
from typing import Any, Optional

import numpy as np

from src.config import Config, device_info

LOGGER = logging.getLogger(__name__)


def time_probe_call(
    probe: Any, activation: np.ndarray, repeats: int
) -> dict[str, float]:
    """Time ``scaler.transform`` + ``decision_function`` on one activation.

    One vector, not a batch: the deployed cost is per response, and batching
    would report an amortised number that no single request ever experiences.

    A short warm-up runs first so that BLAS thread-pool spin-up and the first
    allocation do not land inside the measured samples.

    Args:
        probe: A fitted probe exposing ``score``.
        activation: A single ``(hidden,)`` activation vector.
        repeats: Number of timed repetitions.

    Returns:
        Median, p95, mean, min and max in microseconds, plus the repeat count.
    """
    vector = np.asarray(activation, dtype=np.float32).reshape(1, -1)

    for _ in range(min(50, repeats)):
        probe.score(vector)

    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        start = time.perf_counter()
        probe.score(vector)
        samples[i] = time.perf_counter() - start

    micros = samples * 1e6
    return {
        "median_us": float(np.median(micros)),
        "p95_us": float(np.quantile(micros, 0.95)),
        "mean_us": float(np.mean(micros)),
        "min_us": float(np.min(micros)),
        "max_us": float(np.max(micros)),
        "repeats": int(repeats),
    }


def compare_costs(
    probe_timing: dict[str, float],
    median_generate_seconds: float,
    median_prefill_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Express the probe's cost as a fraction of the costs it rides along with.

    Two denominators are reported. Generation time is the one a user waits for.
    Prefill time is the honest one for the *marginal* cost question, since the
    activation falls out of a forward pass the model was going to run anyway.

    Args:
        probe_timing: Output of :func:`time_probe_call`.
        median_generate_seconds: Median ``generate()`` time per response, from
            the extraction run.
        median_prefill_seconds: Median prefill forward time per response.

    Returns:
        Both costs in milliseconds, the ratios, and the order of magnitude.
    """
    probe_seconds = probe_timing["median_us"] / 1e6
    ratio_generation = (
        probe_seconds / median_generate_seconds if median_generate_seconds else None
    )
    ratio_prefill = (
        probe_seconds / median_prefill_seconds
        if median_prefill_seconds
        else None
    )
    return {
        "probe_median_us": probe_timing["median_us"],
        "probe_p95_us": probe_timing["p95_us"],
        "generation_median_ms": median_generate_seconds * 1e3,
        "prefill_median_ms": (
            median_prefill_seconds * 1e3 if median_prefill_seconds else None
        ),
        "probe_over_generation": ratio_generation,
        "probe_over_prefill": ratio_prefill,
        "orders_of_magnitude": (
            float(-np.log10(ratio_generation)) if ratio_generation else None
        ),
        "adds_a_forward_pass": False,
        "note": (
            "The probe adds no additional forward pass. The activation is a "
            "by-product of the prefill that generation already performs, so the "
            "marginal cost is the scale-and-dot-product timed here."
        ),
    }


def measure(
    probe: Any,
    activation: np.ndarray,
    median_generate_seconds: float,
    median_prefill_seconds: Optional[float],
    config: Config,
) -> dict[str, Any]:
    """Run the full latency measurement and stamp it with the device.

    The device, torch version and quantisation setting travel with the numbers
    because a microsecond figure without them is meaningless (SPEC.md §8).

    Args:
        probe: A fitted probe.
        activation: One real activation vector from the extraction run.
        median_generate_seconds: From ``extract_meta.json``.
        median_prefill_seconds: From ``extract_meta.json``.
        config: Resolved experiment config.

    Returns:
        A JSON-serialisable latency block.
    """
    timing = time_probe_call(probe, activation, config.latency.probe_timing_repeats)
    comparison = compare_costs(
        timing, median_generate_seconds, median_prefill_seconds
    )
    LOGGER.info(
        "probe %.1f us median (p95 %.1f) vs generation %.1f ms -> ratio %.2e",
        timing["median_us"],
        timing["p95_us"],
        comparison["generation_median_ms"],
        comparison["probe_over_generation"]
        if comparison["probe_over_generation"]
        else float("nan"),
    )
    versions: dict[str, Any] = {}
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        versions["torch"] = None
    return {
        "probe_timing": timing,
        "comparison": comparison,
        "device": device_info(),
        "versions": versions,
        "quantization": config.model.quantization,
        "model": config.model.name,
        "hidden_size": int(np.asarray(activation).reshape(1, -1).shape[1]),
    }
