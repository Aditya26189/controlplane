"""Freeze the banking pilot's prompts and measure their distance from the fitted envelope.

``DECISIONS.md`` 090 (corrected), 101. Thin wrapper: parses arguments, calls
``controlplane/``, writes files. No logic (``CLAUDE.md``).

CPU only. Two outputs:

1. **``evalsets/banking-dual-24.draft.json``** -- the 24 prompts, their gold
   answers and each gold answer's provenance, under a content hash. **No
   correctness labels**, because on this set correctness is measured by
   generating an answer and judging it, and a placeholder would be
   indistinguishable from a measurement once it reached an artifact.

2. **``results/pilot_envelope.json``** -- token length and script mix against
   the ``triviaqa-2400-t960`` reference. ``090``'s "cheap de-risking": it does
   not predict whether the probe's *signal* transfers, but a very large input
   distance is an early warning worth having before spending GPU quota.

The GPU pass then generates, judges, extracts and scores. Nothing here produces
a number about the probe.

Usage:
    python scripts/12_pilot_freeze.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from controlplane.config import (  # noqa: E402
    load_config,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from controlplane.evalsets.banking import (  # noqa: E402
    BANKING_PILOT_QUESTIONS,
    build_banking_dual_pilot,
)
from controlplane.extract.model import (  # noqa: E402
    load_tokenizer,
    token_length_summary,
)
from controlplane.evalsets.registry import load_evalset  # noqa: E402

_LOG = logging.getLogger("scripts.12_pilot_freeze")


def _script_mix(text: str) -> dict[str, float]:
    """Fraction of letters by script. Latin vs Devanagari vs everything else.

    The pilot is Hinglish written in Latin script, so this is expected to look
    close to the reference on script and to differ on vocabulary -- which is
    exactly the limit of what a surface metric can tell us, and why ``090``
    calls it an early warning rather than a prediction.
    """
    counts = {"latin": 0, "devanagari": 0, "other": 0}
    for char in text:
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:  # pragma: no cover - unnamed codepoint
            counts["other"] += 1
            continue
        if name.startswith("LATIN"):
            counts["latin"] += 1
        elif name.startswith("DEVANAGARI"):
            counts["devanagari"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _summarise(prompts: list[str]) -> dict:
    """Surface statistics of a prompt set, in units that need no tokenizer.

    **The whitespace count here is a proxy, and it is the wrong unit.** The
    probe reads activations, activations are indexed by model tokens, and
    Qwen2.5 fragments romanised Hindi and digit runs far more aggressively than
    English -- ``Aadhaar 9999 0687 2026 valid hai kya?`` is 7 whitespace tokens
    and 23 model tokens. The real envelope measurement is
    ``model_tokens``, added by :func:`main` when a tokenizer loads. This stays
    because it is free, it is comparable across the two sets, and removing it
    would delete the record of what the earlier artifact actually reported.
    """
    lengths = [len(p.split()) for p in prompts]
    chars = [len(p) for p in prompts]
    mixes = [_script_mix(p) for p in prompts]
    return {
        "n": len(prompts),
        "whitespace_tokens": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        },
        "characters": {
            "mean": statistics.fmean(chars),
            "median": statistics.median(chars),
        },
        "script_mix_mean": {
            k: statistics.fmean([m[k] for m in mixes]) for k in ("latin", "devanagari", "other")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--evalsets-out", default=str(PROJECT_ROOT / "evalsets"),
    )
    parser.add_argument(
        "--out", default=str(PROJECT_ROOT / "results" / "pilot_envelope.json"),
    )
    parser.add_argument(
        "--reference", default="triviaqa-2400-t960",
        help="the envelope the probe was fitted on",
    )
    parser.add_argument(
        "--no-model-tokens", action="store_true",
        help=(
            "skip the model-token measurement. The tokenizer is a ~10 MB "
            "download, no weights and no GPU; skipping leaves the envelope "
            "measured only by a proxy and the artifact says so."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    set_seeds(config.seed)

    draft = build_banking_dual_pilot(seed=config.seed)
    _LOG.info(
        "built %s: %d items over %d questions, hash %s",
        draft.eval_set_id,
        len(draft.items),
        len({i.question_id for i in draft.items}),
        draft.content_hash[:16],
    )

    rot = {}
    for question in BANKING_PILOT_QUESTIONS:
        rot[question.rot_class] = rot.get(question.rot_class, 0) + 1
    checked = sorted({q.gold_checked for q in BANKING_PILOT_QUESTIONS})
    _LOG.info("gold answers by rot class: %s, checked %s", rot, ", ".join(checked))

    draft_path = Path(args.evalsets_out) / f"{draft.eval_set_id}.draft.json"
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(
        json.dumps(draft.to_payload(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _LOG.info("wrote %s", draft_path)

    pilot_prompts = [i.prompt for i in draft.items]
    pilot = _summarise(pilot_prompts)

    reference_path = Path(args.evalsets_out) / f"{args.reference}.json"
    reference = None
    reference_prompts: list[str] = []
    if reference_path.is_file():
        reference_set = load_evalset(reference_path)
        reference_prompts = [i.prompt for i in reference_set.items]
        reference = _summarise(reference_prompts)
        _LOG.info(
            "reference %s: %d items, mean %.1f whitespace tokens",
            args.reference, reference["n"], reference["whitespace_tokens"]["mean"],
        )
    else:
        _LOG.warning(
            "no reference eval set at %s; the distance cannot be computed and is "
            "reported absent rather than as zero", reference_path,
        )

    _LOG.info(
        "pilot: %d items, mean %.1f whitespace tokens, script mix %s",
        pilot["n"],
        pilot["whitespace_tokens"]["mean"],
        {k: round(v, 3) for k, v in pilot["script_mix_mean"].items()},
    )

    whitespace_ratio = None
    if reference is not None:
        whitespace_ratio = (
            pilot["whitespace_tokens"]["mean"] / reference["whitespace_tokens"]["mean"]
        )
        _LOG.info("whitespace proxy ratio pilot/reference: %.3f", whitespace_ratio)

    # ------------------------------------------------------ model tokens (B1)
    # The envelope measurement proper. Tokenizer only -- no weights, no GPU.
    # If it cannot be loaded the fields stay None and carry the reason, because
    # a plausible number in place of an absent measurement is the failure this
    # project exists to catch.
    model_ratio = None
    model_tokens_unavailable = None
    if args.no_model_tokens:
        model_tokens_unavailable = "skipped: --no-model-tokens was passed"
    else:
        try:
            tokenizer = load_tokenizer(config.model.name)
        except Exception as exc:  # noqa: BLE001 - the reason is the artifact
            model_tokens_unavailable = (
                f"tokenizer {config.model.name!r} could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
            _LOG.warning("%s", model_tokens_unavailable)
        else:
            pilot["model_tokens"] = token_length_summary(pilot_prompts, tokenizer)
            _LOG.info(
                "pilot: mean %.1f model tokens, IQR %.1f",
                pilot["model_tokens"]["mean"], pilot["model_tokens"]["iqr"],
            )
            if reference is not None:
                reference["model_tokens"] = token_length_summary(
                    reference_prompts, tokenizer
                )
                model_ratio = (
                    pilot["model_tokens"]["mean"] / reference["model_tokens"]["mean"]
                )
                _LOG.info(
                    "reference: mean %.1f model tokens, IQR %.1f",
                    reference["model_tokens"]["mean"],
                    reference["model_tokens"]["iqr"],
                )
                _LOG.info("MODEL-TOKEN ratio pilot/reference: %.3f", model_ratio)
                if model_ratio > 2.0:
                    _LOG.warning(
                        "model-token ratio %.3f exceeds 2.0: this is a genuine "
                        "envelope finding and it changes what the GPU run is "
                        "testing. Report before proceeding (handoff plan, B2).",
                        model_ratio,
                    )

    payload = {
        "eval_set_id": draft.eval_set_id,
        "draft_content_hash": draft.content_hash,
        "reference_eval_set_id": args.reference,
        "pilot": pilot,
        "reference": reference,
        "length_ratio_model_tokens": model_ratio,
        "length_ratio_whitespace_proxy": whitespace_ratio,
        "model_tokens_unavailable_reason": model_tokens_unavailable,
        "labels": "UNMEASURED - correctness is judged on the generation pass",
        "units": (
            "length_ratio_model_tokens is THE envelope length measurement: the "
            "probe reads activations and activations are indexed by model "
            "tokens. length_ratio_whitespace_proxy is a PROXY for it, retained "
            "because it is free and because it is what the pre-B1 artifact "
            "reported. The two differ because Qwen2.5 fragments romanised "
            "Hindi and digit runs far more aggressively than English, so the "
            "proxy understates the distance on exactly the traffic this pilot "
            "is built from."
        ),
        "interpretation": (
            "Surface distance only. DECISIONS 090 calls this cheap de-risking: "
            "it does not predict whether the probe's signal transfers, and a "
            "small distance here is NOT evidence that it does. The saturation "
            "criterion in 101 (IQR ratio below 0.439) is what decides that, "
            "and it needs the GPU pass."
        ),
        "preregistered_in": "DECISIONS.md 090 (corrected), 101, 102",
    }
    write_json_artifact(Path(args.out), payload, config)
    _LOG.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
