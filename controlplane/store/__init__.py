"""The audit store: an append-only, hash-chained SQLite ledger.

Everything the system asserts ends up here — certificates, warrants issued *and
refused*, validation runs, retention events — in one chain, so the order between
them is part of what the log establishes.

Refusals are stored exactly like issuances. A log recording only the warrants
that were granted would let a refusal be retried quietly until it passed, and
the append-only log is what makes that checkable by someone who was not present.
"""

from .ledger import (
    GENESIS_HASH,
    ChainBreak,
    ChainVerification,
    Ledger,
    LedgerError,
    LedgerRecord,
    RecordKind,
)

__all__ = [
    "GENESIS_HASH",
    "ChainBreak",
    "ChainVerification",
    "Ledger",
    "LedgerError",
    "LedgerRecord",
    "RecordKind",
]
