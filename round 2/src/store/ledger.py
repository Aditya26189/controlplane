"""An append-only, hash-chained SQLite ledger. ``SPEC.md`` §1.5.

One table holds every record — certificates, warrants, validation runs,
retention events — in a single chain:

    self_hash = SHA256(prev_hash || canonical_json(record))

One chain rather than one per record type, because ordering *between* types is
part of what the log has to establish: that a warrant was issued before the
certificate that cited it, and that neither was inserted afterwards. Two chains
could each verify while telling inconsistent stories about the same afternoon.

Tamper-evidence is the point, and it works in two directions. Editing a row's
body breaks that row's own hash. Editing the body *and* recomputing its hash
breaks the next row's ``prev_hash`` instead. Either way
:meth:`Ledger.verify_chain` reports *where*, not merely that.
``test_hash_chain`` performs both attacks with raw SQL and asserts each is
caught at the row it should be.

**What this does not defend against, stated plainly:** an attacker who can
delete every row can also delete the retention event that would have declared
the deletion, and an empty ledger verifies. No self-contained log can do better
— detecting total erasure requires an anchor outside the file (a published head
hash, a second store, a notary). We have not built one, so the honest claim is
tamper-*evidence* against edits and partial deletions, not tamper-proofing.
``DECISIONS.md`` 025.

DPDP Rule 6 shapes the schema: at least one year of retention, and queryable by
session, time range, policy version, detector version, warrant status and
personal-data category. Those six predicates are indexed columns denormalised out
of the record body, because scanning a year of JSON to answer "which requests
touched Aadhaar data?" is a query nobody runs twice.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from ..model import (
    Certificate,
    Warrant,
    WarrantKey,
    WarrantStatus,
    canonical_json,
    chain_hash,
    from_jsonable,
    parse_utc,
    to_jsonable,
    utc_now,
)

__all__ = [
    "ChainBreak",
    "ChainVerification",
    "Ledger",
    "LedgerError",
    "LedgerRecord",
    "RecordKind",
    "GENESIS_HASH",
]

_LOG = logging.getLogger(__name__)

#: Anchor for the first record in a chain. A fixed, obviously-not-a-hash string
#: rather than 64 zeros: zeros are a plausible output of a broken hash function
#: or an uninitialised buffer, and the genesis anchor should never be mistakable
#: for a computed value.
GENESIS_HASH = "GENESIS"

_SCHEMA_VERSION = 1


class RecordKind:
    """The kinds of record the ledger holds.

    Plain string constants rather than an enum: they are written into a column
    that must stay readable by a SQLite client with none of this code, a year
    from now (``SPEC.md`` §1.5 — self-describing enough to interpret without the
    current codebase).
    """

    CERTIFICATE = "certificate"
    WARRANT = "warrant"
    VALIDATION_RUN = "validation_run"
    RETENTION_EVENT = "retention_event"

    ALL = (CERTIFICATE, WARRANT, VALIDATION_RUN, RETENTION_EVENT)


class LedgerError(RuntimeError):
    """Raised when an append would break the ledger's guarantees."""


@dataclasses.dataclass(frozen=True)
class LedgerRecord:
    """One sealed row, as read back out.

    Args:
        seq: Position in the chain, 1-based and gapless.
        kind: One of :class:`RecordKind`.
        record_id: The record's own id, unique within its kind.
        created_at: When it was appended, UTC.
        body: The record as plain data, exactly as hashed.
        prev_hash: The previous row's ``self_hash``.
        self_hash: ``SHA256(prev_hash || canonical_json(body))``.
    """

    seq: int
    kind: str
    record_id: str
    created_at: datetime
    body: dict[str, Any]
    prev_hash: str
    self_hash: str


@dataclasses.dataclass(frozen=True)
class ChainBreak:
    """One inconsistency found while verifying the chain.

    Args:
        seq: Where the break was found.
        record_id: The record at that position.
        reason: What did not match.
        expected: The hash the chain implies.
        found: The hash stored in the row.
    """

    seq: int
    record_id: str
    reason: str
    expected: str
    found: str


@dataclasses.dataclass(frozen=True)
class ChainVerification:
    """The outcome of verifying the whole chain.

    ``first_break_seq`` is the useful field. "The log has been tampered with" is
    an alarm; "row 4,812 no longer matches, and every row after it is
    consequently unverifiable" is an investigation.

    ``anchor_hash`` is what the chain starts from. For an untruncated log that
    is :data:`GENESIS_HASH`. After a logged retention purge it is the hash of
    the last removed record, and a matching ``retention_event`` must exist to
    account for it — otherwise rows were deleted without being declared, which
    is reported as a break like any other.
    """

    ok: bool
    n_records: int
    breaks: tuple[ChainBreak, ...]
    head_hash: str
    anchor_hash: str = GENESIS_HASH

    @property
    def first_break_seq(self) -> Optional[int]:
        """Sequence number of the earliest break, or None if the chain is intact."""
        return self.breaks[0].seq if self.breaks else None

    @property
    def is_truncated(self) -> bool:
        """Whether the chain begins after a declared retention purge."""
        return self.anchor_hash != GENESIS_HASH


class Ledger:
    """Append-only hash-chained store over SQLite.

    Not thread-safe by design for this build: the demo and the pipeline are
    single-writer, and a locking scheme nobody exercised would be a claim about
    concurrency we have not measured.

    Args:
        path: Database file. ``":memory:"`` is accepted for tests.
        retention_days: Minimum retention, from ``config.store.retention_days``.
            Enforced as a floor by :meth:`purge_older_than`, which refuses to
            delete anything younger.
        clock: Source of the current time. Injectable because retention is
            time-dependent behaviour and the alternative way to test a
            400-day floor is to wait 400 days.
    """

    def __init__(
        self,
        path: str | Path,
        retention_days: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if retention_days < 365:
            raise LedgerError(
                f"retention_days must be at least 365 (DPDP Rule 6), got "
                f"{retention_days}"
            )
        self.path = str(path)
        self.retention_days = retention_days
        self._clock = clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        # Foreign keys are unused (one table), but WAL matters: the demo reads
        # the ledger while the pipeline writes to it.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    # -- lifecycle ---------------------------------------------------------- #

    def _create_schema(self) -> None:
        """Create the ledger table and the indexes DPDP Rule 6 requires."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ledger (
                    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind             TEXT    NOT NULL,
                    record_id        TEXT    NOT NULL,
                    created_at       TEXT    NOT NULL,
                    body             TEXT    NOT NULL,
                    prev_hash        TEXT    NOT NULL,
                    self_hash        TEXT    NOT NULL,
                    -- denormalised query columns; see the module docstring
                    session_id       TEXT,
                    policy_version   TEXT,
                    detector_versions TEXT,
                    warrant_status   TEXT,
                    categories       TEXT,
                    detector_id      TEXT,
                    operating_point_id TEXT,
                    eval_set_id      TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(_SCHEMA_VERSION),),
            )
            for column in (
                "session_id",
                "created_at",
                "policy_version",
                "warrant_status",
                "kind",
                "detector_id",
                "eval_set_id",
            ):
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_ledger_{column} "
                    f"ON ledger ({column})"
                )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ledger_identity "
                "ON ledger (kind, record_id)"
            )

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- chain -------------------------------------------------------------- #

    def head_hash(self) -> str:
        """Hash of the most recent record, or the genesis anchor if empty."""
        row = self._conn.execute(
            "SELECT self_hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row["self_hash"] if row else GENESIS_HASH

    def contains(self, kind: str, record_id: str) -> bool:
        """Whether this exact record is already in the ledger.

        Exists because determinism and append-only pull against each other in a
        way that is easy to misread. Warrant ids are content-derived, so a
        re-run of the same validation on the same code produces the *same* id --
        which is the property ``test_determinism`` checks. Appending it again
        would record the same fact twice under two sequence numbers, so callers
        check first and skip.

        The alternative, making append idempotent, would hide the difference
        between "already recorded" and "recorded again", and the second is what
        a re-validation after a code change looks like.
        """
        row = self._conn.execute(
            "SELECT 1 FROM ledger WHERE kind = ? AND record_id = ? LIMIT 1",
            (kind, record_id),
        ).fetchone()
        return row is not None

    def count(self, kind: Optional[str] = None) -> int:
        """Number of records, optionally of one kind."""
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM ledger").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM ledger WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row["n"])

    def _append(
        self,
        kind: str,
        record_id: str,
        body: dict[str, Any],
        columns: Optional[dict[str, Any]] = None,
    ) -> LedgerRecord:
        """Seal a record into the chain and write it.

        The hash is computed over the body *as stored*, so verification can
        recompute it from the row alone without reconstructing the object. That
        matters for the year-later requirement: a verifier needs SQLite and a
        SHA-256, not this codebase.

        Raises:
            LedgerError: If a record with this kind and id already exists.
                Appending a second version of the same id would make "the
                warrant" ambiguous in a log whose whole job is to be unambiguous.
        """
        if kind not in RecordKind.ALL:
            raise LedgerError(f"unknown record kind {kind!r}; known {RecordKind.ALL}")
        existing = self._conn.execute(
            "SELECT seq FROM ledger WHERE kind = ? AND record_id = ?",
            (kind, record_id),
        ).fetchone()
        if existing is not None:
            raise LedgerError(
                f"{kind} {record_id!r} is already in the ledger at seq "
                f"{existing['seq']}. The ledger is append-only: a changed record "
                "is a new record with a new id, never an overwrite of an old one."
            )
        prev = self.head_hash()
        self_hash = chain_hash(prev, body)
        created_at = _as_utc(self._clock())
        payload = canonical_json(body)
        cols = columns or {}
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO ledger (
                    kind, record_id, created_at, body, prev_hash, self_hash,
                    session_id, policy_version, detector_versions, warrant_status,
                    categories, detector_id, operating_point_id, eval_set_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    record_id,
                    created_at.isoformat(),
                    payload,
                    prev,
                    self_hash,
                    cols.get("session_id"),
                    cols.get("policy_version"),
                    cols.get("detector_versions"),
                    cols.get("warrant_status"),
                    cols.get("categories"),
                    cols.get("detector_id"),
                    cols.get("operating_point_id"),
                    cols.get("eval_set_id"),
                ),
            )
        _LOG.debug("ledger append kind=%s id=%s seq=%s", kind, record_id, cursor.lastrowid)
        return LedgerRecord(
            seq=int(cursor.lastrowid),
            kind=kind,
            record_id=record_id,
            created_at=created_at,
            body=body,
            prev_hash=prev,
            self_hash=self_hash,
        )

    def verify_chain(self) -> ChainVerification:
        """Recompute every hash and report where the chain first breaks.

        Three things are checked. Per row: that ``prev_hash`` equals the
        previous row's ``self_hash`` (nothing was inserted, removed or
        reordered), and that ``self_hash`` equals
        ``SHA256(prev_hash || body)`` (the body was not edited). A row edited in
        place fails the second check and every row after it fails the first,
        which is why the report names the *first* break rather than counting.

        Once for the whole chain: that the first row's ``prev_hash`` is either
        the genesis anchor or the hash of the last record removed by a declared
        retention purge. Deleting the head of a log and letting the remainder
        re-anchor itself would otherwise verify cleanly — which would make
        deletion the one edit the chain could not see.

        Returns:
            A :class:`ChainVerification` naming every break found.
        """
        rows = list(
            self._conn.execute(
                "SELECT seq, kind, record_id, body, prev_hash, self_hash "
                "FROM ledger ORDER BY seq"
            )
        )
        if not rows:
            return ChainVerification(
                ok=True, n_records=0, breaks=(), head_hash=GENESIS_HASH
            )

        breaks: list[ChainBreak] = []
        anchor = rows[0]["prev_hash"]
        if anchor != GENESIS_HASH and not self._purge_declared(anchor, rows):
            breaks.append(
                ChainBreak(
                    seq=int(rows[0]["seq"]),
                    record_id=rows[0]["record_id"],
                    reason=(
                        "chain does not start at genesis and no retention_event "
                        "accounts for the removed records"
                    ),
                    expected=GENESIS_HASH,
                    found=anchor,
                )
            )

        expected_prev = anchor
        head = anchor
        for row in rows:
            if row["prev_hash"] != expected_prev:
                breaks.append(
                    ChainBreak(
                        seq=int(row["seq"]),
                        record_id=row["record_id"],
                        reason="prev_hash does not match the preceding record",
                        expected=expected_prev,
                        found=row["prev_hash"],
                    )
                )
            recomputed = chain_hash(row["prev_hash"], json.loads(row["body"]))
            if recomputed != row["self_hash"]:
                breaks.append(
                    ChainBreak(
                        seq=int(row["seq"]),
                        record_id=row["record_id"],
                        reason="record body does not hash to its stored self_hash",
                        expected=recomputed,
                        found=row["self_hash"],
                    )
                )
            expected_prev = row["self_hash"]
            head = row["self_hash"]

        breaks.sort(key=lambda b: b.seq)
        return ChainVerification(
            ok=not breaks,
            n_records=len(rows),
            breaks=tuple(breaks),
            head_hash=head,
            anchor_hash=anchor,
        )

    @staticmethod
    def _purge_declared(anchor: str, rows: list[sqlite3.Row]) -> bool:
        """Whether a retention_event in the surviving chain explains the anchor.

        The purge record is appended after the deletion, so it survives inside
        the shortened chain and names the hash of the last record it removed.
        Matching that against the anchor is what keeps a declared purge
        distinguishable from a quiet one.
        """
        for row in rows:
            if row["kind"] != RecordKind.RETENTION_EVENT:
                continue
            body = json.loads(row["body"])
            if body.get("last_hash_removed") == anchor:
                return True
        return False

    # -- writing ------------------------------------------------------------ #

    def append_certificate(self, certificate: Certificate) -> Certificate:
        """Seal a certificate into the chain and return the sealed record.

        The caller gets back a new frozen record carrying its chain position.
        The record passed in is never mutated, because it may already have been
        handed to something else.

        Args:
            certificate: An unsealed certificate.

        Returns:
            The same certificate with ``prev_certificate_hash`` and ``self_hash``
            assigned.

        Raises:
            LedgerError: If the certificate arrives already sealed — that means
                it came from the ledger, and re-appending it would put the same
                decision in the log twice under two different hashes.
        """
        if certificate.is_sealed:
            raise LedgerError(
                f"certificate {certificate.certificate_id} is already sealed; it "
                "has been appended once already"
            )
        body = to_jsonable(certificate)
        record = self._append(
            RecordKind.CERTIFICATE,
            certificate.certificate_id,
            body,
            columns={
                "session_id": certificate.session_id,
                "policy_version": certificate.resolution.policy_version,
                "detector_versions": ",".join(certificate.detector_versions()),
                "warrant_status": certificate.weakest_warrant_status.value,
                "categories": ",".join(certificate.categories_accessed()),
                "eval_set_id": certificate.envelope_match.envelope_id,
            },
        )
        return certificate.sealed_with(record.prev_hash, record.self_hash)

    def append_warrant(self, warrant: Warrant) -> LedgerRecord:
        """Seal a warrant — issued or refused — into the chain.

        Refusals are appended exactly like issuances. A log that recorded only
        the warrants that were granted would let a refusal be quietly retried
        until it passed, which is the behaviour ``CLAUDE.md`` invariant 2 of the
        Round 1 contract called out and the reason the append-only log exists at
        all.
        """
        return self._append(
            RecordKind.WARRANT,
            warrant.warrant_id,
            to_jsonable(warrant),
            columns={
                "warrant_status": warrant.status.value,
                "detector_id": warrant.detector_id,
                "detector_versions": f"{warrant.detector_id}@{warrant.detector_version}",
                "operating_point_id": warrant.operating_point.operating_point_id,
                "eval_set_id": warrant.eval_set_id,
            },
        )

    def append_validation_run(
        self, run_id: str, body: dict[str, Any], eval_set_id: str, detector_id: str
    ) -> LedgerRecord:
        """Seal a validation run's summary into the chain.

        Every scoring of a test set is logged here, which is how "test is scored
        once per validation run, and every run is published" becomes checkable by
        someone who was not present for any of them.
        """
        return self._append(
            RecordKind.VALIDATION_RUN,
            run_id,
            body,
            columns={"eval_set_id": eval_set_id, "detector_id": detector_id},
        )

    # -- reading ------------------------------------------------------------ #

    def _row_to_record(self, row: sqlite3.Row) -> LedgerRecord:
        return LedgerRecord(
            seq=int(row["seq"]),
            kind=row["kind"],
            record_id=row["record_id"],
            created_at=parse_utc(row["created_at"]),
            body=json.loads(row["body"]),
            prev_hash=row["prev_hash"],
            self_hash=row["self_hash"],
        )

    def get_certificate(self, certificate_id: str) -> Certificate:
        """Read one certificate back, fully typed.

        Raises:
            LedgerError: If no such certificate exists.
        """
        row = self._conn.execute(
            "SELECT * FROM ledger WHERE kind = ? AND record_id = ?",
            (RecordKind.CERTIFICATE, certificate_id),
        ).fetchone()
        if row is None:
            raise LedgerError(f"no certificate {certificate_id!r} in the ledger")
        return from_jsonable(Certificate, json.loads(row["body"])).sealed_with(
            row["prev_hash"], row["self_hash"]
        )

    def get_warrant(self, warrant_id: str) -> Warrant:
        """Read one warrant back, fully typed.

        Raises:
            LedgerError: If no such warrant exists.
        """
        row = self._conn.execute(
            "SELECT body FROM ledger WHERE kind = ? AND record_id = ?",
            (RecordKind.WARRANT, warrant_id),
        ).fetchone()
        if row is None:
            raise LedgerError(f"no warrant {warrant_id!r} in the ledger")
        return from_jsonable(Warrant, json.loads(row["body"]))

    def warrants_for_key(self, key: WarrantKey) -> tuple[Warrant, ...]:
        """Every warrant ever filed under one (detector, operating point, envelope).

        Returned oldest-first. More than one is normal and expected:
        revalidation appends rather than replaces, so the history of a cell is
        readable — including the refusals.
        """
        rows = self._conn.execute(
            """
            SELECT body FROM ledger
            WHERE kind = ? AND detector_id = ? AND operating_point_id = ?
              AND eval_set_id = ?
            ORDER BY seq
            """,
            (
                RecordKind.WARRANT,
                key.detector_id,
                key.operating_point_id,
                key.eval_set_id,
            ),
        ).fetchall()
        return tuple(from_jsonable(Warrant, json.loads(r["body"])) for r in rows)

    def latest_warrant(self, key: WarrantKey) -> Optional[Warrant]:
        """The most recent warrant for a key, or None if the cell is empty.

        None means ``UNVALIDATED`` (``DECISIONS.md`` 024): the cell has never
        been measured. Returned as an absence rather than as a record so that no
        caller can read bounds off it by accident.
        """
        warrants = self.warrants_for_key(key)
        return warrants[-1] if warrants else None

    def certificates_for_session(self, session_id: str) -> tuple[Certificate, ...]:
        """Every certificate in one session, oldest first. DPDP Rule 6."""
        rows = self._conn.execute(
            "SELECT * FROM ledger WHERE kind = ? AND session_id = ? ORDER BY seq",
            (RecordKind.CERTIFICATE, session_id),
        ).fetchall()
        return tuple(
            from_jsonable(Certificate, json.loads(r["body"])).sealed_with(
                r["prev_hash"], r["self_hash"]
            )
            for r in rows
        )

    def query(
        self,
        *,
        kind: Optional[str] = None,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        policy_version: Optional[str] = None,
        detector_version: Optional[str] = None,
        warrant_status: Optional[WarrantStatus] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> tuple[LedgerRecord, ...]:
        """The DPDP Rule 6 query surface, in one method.

        Every predicate the rule names is here: session, time range, policy
        version, detector version, warrant status, and personal-data category
        accessed. They compose, because the questions asked after an incident
        are conjunctions — *"which sessions in March, under policy 3.1, touched
        Aadhaar data using the detector version we have just withdrawn?"*

        Args:
            kind: Restrict to one record kind.
            session_id: Exact session match.
            since: Inclusive lower bound on ``created_at``.
            until: Exclusive upper bound on ``created_at``.
            policy_version: Exact policy version match.
            detector_version: Substring match against the denormalised
                ``detector_id@version`` list, since one certificate can cite
                several detectors.
            warrant_status: Weakest warrant status for certificates, or the
                warrant's own status for warrant records.
            category: Personal-data or finding category accessed.
            limit: Maximum rows.

        Returns:
            Matching records, oldest first.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(_as_utc(since).isoformat())
        if until is not None:
            clauses.append("created_at < ?")
            params.append(_as_utc(until).isoformat())
        if policy_version is not None:
            clauses.append("policy_version = ?")
            params.append(policy_version)
        if detector_version is not None:
            clauses.append("detector_versions LIKE ?")
            params.append(f"%{detector_version}%")
        if warrant_status is not None:
            clauses.append("warrant_status = ?")
            params.append(warrant_status.value)
        if category is not None:
            clauses.append("categories LIKE ?")
            params.append(f"%{category}%")
        sql = "SELECT * FROM ledger"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return tuple(self._row_to_record(r) for r in self._conn.execute(sql, params))

    def iter_records(self) -> Iterator[LedgerRecord]:
        """Walk the whole chain oldest-first, for verification and export."""
        for row in self._conn.execute("SELECT * FROM ledger ORDER BY seq"):
            yield self._row_to_record(row)

    # -- retention ---------------------------------------------------------- #

    def retention_floor(self, now: Optional[datetime] = None) -> datetime:
        """The oldest timestamp that must still be retained.

        DPDP Rule 6 sets a *minimum* retention, so this is a floor on deletion,
        not a schedule for it. Nothing older is deleted automatically either —
        purging is an explicit, logged act.
        """
        return _as_utc(now or self._clock()) - timedelta(days=self.retention_days)

    def purge_older_than(
        self, cutoff: datetime, *, now: Optional[datetime] = None, dry_run: bool = True
    ) -> dict[str, Any]:
        """Delete records older than ``cutoff``, refusing to breach the floor.

        Deleting from a hash chain breaks it, and pretending otherwise would be
        worse than not purging: the chain would verify over a log that no longer
        contains what it claims to. So a purge appends a ``retention_event``
        record naming the seq range and hash range removed, and verification
        treats the chain as re-anchored at that event. The log stays honest about
        having been shortened.

        Args:
            cutoff: Delete records created strictly before this.
            now: Current time, injectable for tests.
            dry_run: When true (the default) nothing is deleted and the return
                value describes what would be.

        Returns:
            A summary: how many records, which seq range, and the retention floor
            that was checked.

        Raises:
            LedgerError: If ``cutoff`` is newer than the retention floor.
        """
        cutoff = _as_utc(cutoff)
        floor = self.retention_floor(now)
        if cutoff > floor:
            raise LedgerError(
                f"refusing to purge back to {cutoff.isoformat()}: the retention "
                f"floor is {floor.isoformat()} ({self.retention_days} days). DPDP "
                "Rule 6 sets a minimum retention and this store will not breach it."
            )
        rows = self._conn.execute(
            "SELECT seq, self_hash FROM ledger WHERE created_at < ? ORDER BY seq",
            (cutoff.isoformat(),),
        ).fetchall()
        summary: dict[str, Any] = {
            "cutoff": cutoff.isoformat(),
            "retention_floor": floor.isoformat(),
            "retention_days": self.retention_days,
            "n_records": len(rows),
            "first_seq": int(rows[0]["seq"]) if rows else None,
            "last_seq": int(rows[-1]["seq"]) if rows else None,
            "last_hash_removed": rows[-1]["self_hash"] if rows else None,
            "dry_run": dry_run,
        }
        if dry_run or not rows:
            return summary
        with self._conn:
            self._conn.execute(
                "DELETE FROM ledger WHERE created_at < ?", (cutoff.isoformat(),)
            )
        self._append(
            RecordKind.RETENTION_EVENT,
            f"purge-{cutoff.date().isoformat()}-{summary['last_seq']}",
            dict(summary, dry_run=False),
        )
        _LOG.info(
            "purged %d records older than %s; chain re-anchored",
            summary["n_records"],
            cutoff.isoformat(),
        )
        return summary


def _as_utc(value: datetime) -> datetime:
    """Coerce to UTC, refusing naive datetimes.

    A retention boundary that means a different instant depending on where it is
    evaluated is a compliance bug, not a convenience.
    """
    if value.tzinfo is None:
        raise LedgerError(
            f"timestamp {value!r} is naive; retention and query boundaries carry "
            "an explicit UTC offset"
        )
    return value.astimezone(timezone.utc)
