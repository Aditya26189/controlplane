"""Freezing eval sets to disk, and the manifest that registers them.

``CLAUDE.md`` invariant 9: eval sets are frozen and content-hashed, and changing
one creates a new id. This module is where that becomes a file on disk rather
than a property of an in-memory object.

The manifest is the registry a reviewer reads first: every set, its hash, its
size, its base rate, whether it is measured or synthetic, and a one-line
statement of how it was built. :func:`verify_manifest` rebuilds nothing but
re-hashes everything, so a set edited after registration is caught rather than
trusted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from ..validation.evalsets import EvalItem, EvalSet, EvalSetError

__all__ = [
    "MANIFEST_NAME",
    "load_evalset",
    "read_manifest",
    "save_evalset",
    "verify_manifest",
    "write_manifest",
]

_LOG = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def save_evalset(evalset: EvalSet, directory: str | Path) -> Path:
    """Write one eval set as JSON, named by its id.

    The file records the content hash it had when written. :func:`load_evalset`
    recomputes and compares, so an edit to the file is caught on the next read
    rather than silently producing a set with a different identity than the
    warrants pointing at it.

    Args:
        evalset: The set to freeze.
        directory: Where to write it.

    Returns:
        The path written.
    """
    out = Path(directory) / f"{evalset.eval_set_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "eval_set_id": evalset.eval_set_id,
        "content_hash": evalset.content_hash,
        "envelope_id": evalset.envelope_id,
        "data_source": evalset.data_source,
        "n_items": len(evalset),
        "base_rate": evalset.base_rate,
        "construction": evalset.construction,
        "items": [
            {
                "item_id": item.item_id,
                "question_id": item.question_id,
                "prompt": item.prompt,
                "response": item.response,
                "label": item.label,
                "split": item.split,
                "meta": item.meta,
            }
            for item in evalset.items
        ],
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _LOG.info("wrote %s (%d items, %s)", out, len(evalset), evalset.envelope_id)
    return out


def load_evalset(path: str | Path) -> EvalSet:
    """Read a frozen eval set back, checking its hash.

    Raises:
        EvalSetError: If the file's contents no longer hash to the value it
            records. That means the file was edited after it was frozen, and
            every warrant keyed to the recorded hash describes different data.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evalset = EvalSet(
        eval_set_id=data["eval_set_id"],
        items=tuple(
            EvalItem(
                item_id=item["item_id"],
                question_id=item["question_id"],
                prompt=item["prompt"],
                response=item["response"],
                label=item["label"],
                split=item.get("split"),
                meta=item.get("meta", {}),
            )
            for item in data["items"]
        ),
        data_source=data["data_source"],
        construction=data["construction"],
    )
    recorded = data.get("content_hash")
    if recorded and recorded != evalset.content_hash:
        raise EvalSetError(
            f"{path} records content hash {recorded[:16]} but its contents now "
            f"hash to {evalset.content_hash[:16]}. The file was edited after it "
            "was frozen; every warrant keyed to the recorded hash describes "
            "different data. Rebuild the set rather than editing it in place "
            "(CLAUDE.md invariant 9)."
        )
    return evalset


def write_manifest(
    evalsets: Iterable[EvalSet], directory: str | Path, *, extra: Optional[dict] = None
) -> Path:
    """Write the registry a reviewer reads first.

    Args:
        evalsets: The sets to register.
        directory: Where the manifest and the sets live.
        extra: Anything else worth recording. Must itself be deterministic --
            anything carrying a timestamp or a run id belongs in ``results/``.

    Returns:
        The manifest path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for evalset in evalsets:
        labels = evalset.labels
        entries.append(
            {
                "eval_set_id": evalset.eval_set_id,
                "content_hash": evalset.content_hash,
                "envelope_id": evalset.envelope_id,
                "data_source": evalset.data_source,
                "n_items": len(evalset),
                "n_positive": int(labels.sum()),
                "n_negative": int(labels.size - labels.sum()),
                "base_rate": evalset.base_rate,
                "single_class": bool(len(set(labels.tolist())) < 2),
                "label_meaning": evalset.construction.get("label_meaning", ""),
                "method": evalset.construction.get("method", ""),
                "llm_generated": evalset.construction.get("llm_generated"),
                "file": f"{evalset.eval_set_id}.json",
            }
        )
    # Deterministic by construction: no timestamp, no provenance block, nothing
    # that changes between two runs at one seed. The manifest is a registry of
    # *content*, and a registry that rewrites itself on every build makes the
    # working tree permanently dirty -- which drains the dirty flag of meaning
    # and breaks the clean-clone reproduction the definition of done requires.
    # When the run happened belongs in results/, not in the frozen registry.
    manifest = {
        "n_sets": len(entries),
        "sets": entries,
    }
    if extra:
        manifest.update(extra)
    out = directory / MANIFEST_NAME
    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _LOG.info("wrote %s registering %d sets", out, len(entries))
    return out


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """Read the manifest.

    Raises:
        FileNotFoundError: Naming the script that builds it, since the usual
            cause is not having run it.
    """
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_build_evalsets.py to build and "
            "register the evaluation sets."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(directory: str | Path) -> list[str]:
    """Re-hash every registered set and report any that no longer match.

    The check that makes "frozen" mean something. Reading a manifest tells you
    what was registered; re-hashing tells you whether it is still what was
    registered.

    Args:
        directory: Where the manifest and sets live.

    Returns:
        A list of human-readable problems. Empty means every set matches.
    """
    directory = Path(directory)
    manifest = read_manifest(directory)
    problems: list[str] = []
    for entry in manifest["sets"]:
        path = directory / entry["file"]
        if not path.is_file():
            problems.append(f"{entry['eval_set_id']}: {entry['file']} is missing")
            continue
        try:
            evalset = load_evalset(path)
        except EvalSetError as exc:
            problems.append(f"{entry['eval_set_id']}: {exc}")
            continue
        if evalset.content_hash != entry["content_hash"]:
            problems.append(
                f"{entry['eval_set_id']}: manifest records "
                f"{entry['content_hash'][:16]} but the file hashes to "
                f"{evalset.content_hash[:16]}"
            )
        if len(evalset) != entry["n_items"]:
            problems.append(
                f"{entry['eval_set_id']}: manifest records {entry['n_items']} items "
                f"but the file holds {len(evalset)}"
            )
    return problems
