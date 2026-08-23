"""Typed JSON round-tripping and canonical hashing for the record types.

Two jobs, and the second is why this is not just ``dataclasses.asdict``:

1. **Round-trip.** A record written to the store in 2026 has to be readable in
   2027 without the code that wrote it (``SPEC.md`` §1.5, DPDP Rule 6). So the
   encoding is plain JSON with no pickling and no type tags, and decoding is
   driven by the target class's annotations.
2. **Canonical form.** The hash chain is
   ``self_hash = SHA256(prev_hash || canonical_json(record))``. Two processes
   must agree byte-for-byte on ``canonical_json`` of the same record or the
   chain fails to verify for a reason that has nothing to do with tampering.

Decoding is strict for the same reason config loading is: a record that decodes
with a field silently missing is worse than one that fails to decode, because
the resulting object looks usable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

__all__ = [
    "SerdeError",
    "canonical_json",
    "chain_hash",
    "content_hash",
    "from_jsonable",
    "parse_utc",
    "to_jsonable",
    "utc_now",
]

T = TypeVar("T")


class SerdeError(ValueError):
    """Raised when a record cannot be encoded or decoded faithfully."""


def utc_now() -> datetime:
    """Current time, UTC, whole seconds.

    Whole seconds because timestamps are hashed into the chain and rendered into
    documents; sub-second precision buys nothing here and gives two
    representations of the same instant, which is one more than a canonical form
    can have.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_utc(text: str) -> datetime:
    """Parse an ISO-8601 timestamp, requiring an explicit UTC offset.

    Naive datetimes are rejected rather than assumed local: a record whose
    timestamp means a different instant depending on where it is read is not an
    audit record.
    """
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise SerdeError(
            f"timestamp {text!r} has no timezone. Audit records carry an explicit "
            "UTC offset; a naive timestamp means a different instant depending on "
            "where it is read."
        )
    return parsed.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def to_jsonable(value: Any) -> Any:
    """Convert a record (or any part of one) to plain JSON-compatible data.

    Enums become their ``value``, datetimes become ISO-8601 with an explicit
    offset, tuples become lists, and dataclasses become mappings in field order.
    Nothing carries a type tag: the class being decoded into supplies the types.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise SerdeError(
                "refusing to encode a naive datetime; audit records carry an "
                "explicit UTC offset"
            )
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise SerdeError(
        f"{type(value).__name__} has no JSON encoding. Record types are built "
        "from primitives, enums, datetimes, tuples, dicts and other records; "
        "anything else would not survive being read back a year from now."
    )


def canonical_json(value: Any) -> str:
    """Render a record to the exact string the hash chain is taken over.

    Sorted keys and no whitespace, so two processes that agree on the record
    agree on the bytes. ``ensure_ascii=False`` keeps Devanagari and other
    non-Latin text as itself rather than as escapes — the Hinglish eval set
    would otherwise hash differently depending on the encoder's defaults.
    """
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(value: Any) -> str:
    """SHA-256 of a record's canonical rendering, as 64 hex characters.

    Used for eval-set identity (``CLAUDE.md`` invariant 9: changing a set
    creates a new id) and as the payload half of the chain hash.
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def chain_hash(prev_hash: str, record: Any) -> str:
    """``SHA256(prev_hash || canonical_json(record))`` — ``SPEC.md`` §1.5.

    Concatenating the previous hash is what makes the chain a chain: editing any
    record changes its own hash, so every hash after it no longer matches the
    value stored beside it, and the break is locatable rather than merely
    detectable.

    Args:
        prev_hash: The previous record's ``self_hash``, or the genesis constant
            for the first record.
        record: The record being sealed.

    Returns:
        64 hex characters.
    """
    payload = canonical_json(record).encode("utf-8")
    return hashlib.sha256(prev_hash.encode("utf-8") + payload).hexdigest()


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def _is_optional(typ: Any) -> bool:
    return get_origin(typ) in (Union, UnionType) and type(None) in get_args(typ)


def _unwrap_optional(typ: Any) -> Any:
    args = [a for a in get_args(typ) if a is not type(None)]
    if len(args) != 1:
        raise SerdeError(f"unsupported union in a record type: {typ!r}")
    return args[0]


def _decode(value: Any, typ: Any, path: str) -> Any:
    """Decode one value against its annotation, naming its path on failure."""
    if _is_optional(typ):
        if value is None:
            return None
        return _decode(value, _unwrap_optional(typ), path)
    if is_dataclass(typ):
        return from_jsonable(typ, value, path)
    if isinstance(typ, type) and issubclass(typ, Enum):
        try:
            return typ(value)
        except ValueError as exc:
            raise SerdeError(
                f"{path}: {value!r} is not a member of {typ.__name__}; known "
                f"members are {[m.value for m in typ]}"
            ) from exc
    if typ is datetime:
        if not isinstance(value, str):
            raise SerdeError(f"{path}: expected an ISO-8601 string, got {value!r}")
        return parse_utc(value)
    origin = get_origin(typ)
    if origin is tuple:
        args = get_args(typ)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise SerdeError(
                f"{path}: record sequences are declared as tuple[X, ...]; got {typ!r}"
            )
        if not isinstance(value, (list, tuple)):
            raise SerdeError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_decode(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        key_type, val_type = get_args(typ)
        if key_type is not str:
            raise SerdeError(f"{path}: record mappings must be keyed by string")
        if not isinstance(value, dict):
            raise SerdeError(f"{path}: expected a mapping, got {type(value).__name__}")
        return {k: _decode(v, val_type, f"{path}.{k}") for k, v in value.items()}
    if typ is Any:
        return value
    if typ is bool:
        if not isinstance(value, bool):
            raise SerdeError(f"{path}: expected a boolean, got {value!r}")
        return value
    if typ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SerdeError(f"{path}: expected an integer, got {value!r}")
        return value
    if typ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SerdeError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if typ is str:
        if not isinstance(value, str):
            raise SerdeError(f"{path}: expected a string, got {value!r}")
        return value
    raise SerdeError(f"{path}: unsupported field type {typ!r}")


def from_jsonable(cls: type[T], data: Any, path: str = "") -> T:
    """Rebuild a record from plain data, driven by the class's annotations.

    Strict about unknown and missing keys. A record that decodes with a field
    silently defaulted is worse than one that fails to decode, because the
    object it produces looks usable and its provenance is wrong.

    Args:
        cls: The dataclass to build.
        data: Plain data as produced by :func:`to_jsonable`.
        path: Dotted path used in error messages.

    Returns:
        An instance of ``cls``, with every ``__post_init__`` assertion applied —
        so a record that violates an invariant cannot be read back into a valid
        object either.

    Raises:
        SerdeError: On a type mismatch, an unknown key, or a missing key.
    """
    where = path or cls.__name__
    if not is_dataclass(cls):
        raise SerdeError(f"{where}: {cls!r} is not a record type")
    if not isinstance(data, dict):
        raise SerdeError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    field_map = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(field_map))
    if unknown:
        raise SerdeError(
            f"{where}: unknown field(s) {unknown}; known fields are "
            f"{sorted(field_map)}"
        )
    missing = sorted(
        name
        for name, f in field_map.items()
        if name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise SerdeError(f"{where}: missing field(s) {missing}")
    kwargs = {
        name: _decode(data[name], hints[name], f"{where}.{name}")
        for name in field_map
        if name in data
    }
    return cls(**kwargs)
