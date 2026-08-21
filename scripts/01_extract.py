"""Stage 1+3: prepare splits, extract question-time activations, label.

Parses arguments, calls ``src/``, writes files. No logic lives here
(CLAUDE.md, Coding standards).

Outputs:
    results/splits.parquet      the split assignment every later stage reads
    results/data_stats.json     rows loaded, dedup count, split sizes
    results/activations.npz     fp16 activations keyed by layer
    results/labels.parquet      completions, correctness, labels
    results/extract_meta.json   equivalence check, timings, base rates

Typical use::

    python scripts/01_extract.py --data-only                 # CPU, no model
    python scripts/01_extract.py --limit 20 --dry-run        # pre-flight
    python scripts/01_extract.py                             # the full run
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    load_config,
    set_seeds,
    setup_logging,
    write_json_artifact,
)
from src.data import prepare_dataset, save_splits  # noqa: E402
from src.extract import (  # noqa: E402
    assert_base_rate,
    base_rate_summary,
    check_left_padding_equivalence,
    run_extraction,
    save_activations,
    select_equivalence_prompts,
)
from src.model import build_prompts, describe_model, load_model_and_tokenizer, resolve_layers  # noqa: E402

LOGGER = logging.getLogger("01_extract")

EQUIVALENCE_BATCH = 4
EQUIVALENCE_TOLERANCE = 1e-2
PREVIEW_COMPLETIONS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="override data.n_examples (used for the pre-flight smoke run)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="write splits and data stats, then stop; needs no GPU and no model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run everything but write no artifacts",
    )
    parser.add_argument(
        "--quantization",
        choices=["nf4", "none"],
        default=None,
        help="override model.quantization (use 'none' on CPU)",
    )
    parser.add_argument("--model", default=None, help="override model.name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    overrides: dict[str, object] = {}
    if args.limit is not None:
        overrides["data.n_examples"] = args.limit
    if args.quantization is not None:
        overrides["model.quantization"] = args.quantization
    if args.model is not None:
        overrides["model.name"] = args.model

    config = load_config(args.config, overrides or None)
    set_seeds(config.seed)
    LOGGER.info("config hash %s, seed %d", config.config_hash, config.seed)

    frame, data_stats = prepare_dataset(config)
    if not args.dry_run:
        save_splits(frame, config.paths.splits)
        write_json_artifact(
            config.results_path("data_stats.json"), {"data": data_stats}, config
        )
    if args.data_only:
        LOGGER.info("--data-only: stopping before the model is loaded")
        return 0

    model, tokenizer = load_model_and_tokenizer(config)
    layers = resolve_layers(model, config)
    model_info = describe_model(model, tokenizer, config)
    LOGGER.info(
        "model: %d layers, hidden %d, probing %s",
        model_info["num_hidden_layers"],
        model_info["hidden_size"],
        model_info["probe_layers"],
    )

    # TASKS.md Stage 3, item 1: the highest-value check in the repo. Runs before
    # the expensive loop so a padding fault costs seconds rather than an hour.
    # Spanning the length distribution rather than taking the first few: the
    # check is only as sensitive as the amount of padding in its batch.
    all_prompts = build_prompts(tokenizer, frame["question"].tolist(), config)
    sample = select_equivalence_prompts(tokenizer, all_prompts, EQUIVALENCE_BATCH)
    equivalence = check_left_padding_equivalence(
        model,
        tokenizer,
        sample,
        layers,
        tolerance=EQUIVALENCE_TOLERANCE,
    )
    LOGGER.info("equivalence check passed: max deviation %.3e", equivalence["max_deviation"])

    acts, labelled, extract_meta = run_extraction(model, tokenizer, frame, config, layers)

    LOGGER.info("first %d completions:", PREVIEW_COMPLETIONS)
    preview = labelled.head(PREVIEW_COMPLETIONS)
    for _, row in preview.iterrows():
        LOGGER.info(
            "  [%s] Q: %s | A: %s | gold: %s",
            "OK " if row["correct"] else "ERR",
            row["question"][:70],
            row["completion"][:70].replace("\n", " "),
            row["answer_value"][:40],
        )

    base_rates = base_rate_summary(labelled)

    if args.dry_run:
        LOGGER.info("--dry-run: no artifacts written")
        assert_base_rate(labelled, config)
        return 0

    # Artifacts are written BEFORE the base-rate gate fires. The gate is a
    # judgement about the labels, not about the activations, and the activations
    # cost 40-70 minutes of GPU. Throwing them away would mean re-extracting to
    # investigate why the labels looked wrong -- so persist, then judge.
    save_activations(acts, labelled["question_id"].tolist(), config.paths.activations)
    labelled.to_parquet(config.paths.labels, index=False)
    LOGGER.info("wrote %s (%d rows)", config.paths.labels, len(labelled))

    write_json_artifact(
        config.results_path("extract_meta.json"),
        {
            "data": data_stats,
            "model": model_info,
            "equivalence_check": equivalence,
            "extraction": extract_meta,
            "base_rates": base_rates,
            "base_rate_by_split": {
                name: {
                    "n": int((labelled["split"] == name).sum()),
                    "base_rate_incorrect": float(
                        labelled.loc[labelled["split"] == name, "label"].mean()
                    ),
                }
                for name in ("train", "val", "test")
            },
        },
        config,
    )
    LOGGER.info(
        "stage 3 complete: %d examples in %.1f s (%.2f ex/s)",
        extract_meta["n_examples"],
        extract_meta["total_seconds"],
        extract_meta["examples_per_second"],
    )

    # Now the gate. Everything above is already on disk, so if this raises the
    # run is stopped for inspection rather than lost (TASKS.md Stage 3 gate).
    assert_base_rate(labelled, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
