"""Ledger behaviour, including the ``test_hash_chain`` gate test.

The chain test mutates a row with raw SQL rather than through any API. That is
the point: the guarantee is about what a database file can be made to say, so
the attack has to bypass the code that would refuse it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from controlplane.model import (
    Category,
    WarrantKey,
    WarrantStatus,
    canonical_json,
    chain_hash,
    utc_now,
)
from controlplane.store import GENESIS_HASH, Ledger, LedgerError, RecordKind

from .factories import (
    failing_controls,
    make_certificate,
    make_finding,
    make_warrant,
)


@pytest.fixture()
def ledger(tmp_path: Path) -> Ledger:
    """A ledger on disk, so the chain test can reach it with raw SQL."""
    store = Ledger(tmp_path / "controlplane.db", retention_days=400)
    yield store
    store.close()


# --------------------------------------------------------------------------- #
# Gate test: mutating a row breaks the chain, demonstrably
# --------------------------------------------------------------------------- #


def test_hash_chain(ledger: Ledger, tmp_path: Path) -> None:
    """Editing a stored record breaks the chain and names where.

    ``SPEC.md`` §1.5. The mutation goes in through raw SQL because the claim is
    about what the database file can be made to say — an attacker with write
    access does not politely use the API that would refuse them.
    """
    warrant = make_warrant()
    ledger.append_warrant(warrant)
    for i in range(4):
        ledger.append_certificate(
            make_certificate(certificate_id=f"C-{i}", request_id=f"R-{i}", warrant=warrant)
        )

    before = ledger.verify_chain()
    assert before.ok, before.breaks
    assert before.n_records == 5
    assert before.anchor_hash == GENESIS_HASH
    assert before.first_break_seq is None

    # Tamper: change the action recorded on the third certificate from ALLOW to
    # BLOCK, exactly as someone rewriting history after an incident would.
    conn = sqlite3.connect(ledger.path)
    row = conn.execute("SELECT body FROM ledger WHERE seq = 3").fetchone()
    body = json.loads(row[0])
    assert body["resolution"]["action"] == "ALLOW"
    body["resolution"]["action"] = "BLOCK"
    conn.execute(
        "UPDATE ledger SET body = ? WHERE seq = 3", (canonical_json(body),)
    )
    conn.commit()
    conn.close()

    after = ledger.verify_chain()
    assert not after.ok
    assert after.first_break_seq == 3
    assert any("does not hash to its stored self_hash" in b.reason for b in after.breaks)

    # A more careful attacker recomputes the edited row's hash. That repairs the
    # body check and breaks the *next* row's prev_hash instead, because seq 4
    # still records the hash seq 3 used to have. The two checks cover each
    # other, which is why both exist.
    conn = sqlite3.connect(ledger.path)
    prev_hash, = conn.execute("SELECT prev_hash FROM ledger WHERE seq = 3").fetchone()
    conn.execute(
        "UPDATE ledger SET self_hash = ? WHERE seq = 3",
        (chain_hash(prev_hash, body),),
    )
    conn.commit()
    conn.close()

    repaired = ledger.verify_chain()
    assert not repaired.ok
    assert repaired.first_break_seq == 4
    assert any(
        "prev_hash does not match the preceding record" in b.reason
        for b in repaired.breaks
    )


def test_deleting_the_head_without_declaring_it_is_detected(ledger: Ledger) -> None:
    """Silent truncation is the one edit a naive chain cannot see.

    Delete the first rows and the remainder re-anchors and verifies cleanly
    unless the anchor itself is checked, so it is.
    """
    warrant = make_warrant()
    ledger.append_warrant(warrant)
    for i in range(3):
        ledger.append_certificate(
            make_certificate(certificate_id=f"C-{i}", request_id=f"R-{i}", warrant=warrant)
        )
    assert ledger.verify_chain().ok

    conn = sqlite3.connect(ledger.path)
    conn.execute("DELETE FROM ledger WHERE seq <= 2")
    conn.commit()
    conn.close()

    after = ledger.verify_chain()
    assert not after.ok
    assert after.is_truncated
    assert any("no retention_event accounts" in b.reason for b in after.breaks)


def test_empty_ledger_verifies(ledger: Ledger) -> None:
    result = ledger.verify_chain()
    assert result.ok and result.n_records == 0
    assert result.head_hash == GENESIS_HASH


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #


def test_certificate_round_trips_through_the_store(ledger: Ledger) -> None:
    """The Phase 1 gate: a certificate round-trips."""
    warrant = make_warrant()
    ledger.append_warrant(warrant)
    original = make_certificate(warrant=warrant)
    sealed = ledger.append_certificate(original)

    assert sealed.is_sealed
    assert sealed.prev_certificate_hash != sealed.self_hash
    assert sealed.unsealed() == original

    restored = ledger.get_certificate(original.certificate_id)
    assert restored == sealed
    assert restored.claimed_bounds["recall"]["ci_low"] < restored.claimed_bounds["recall"]["value"]
    assert restored.findings[0].detector_id == "probe-qwen2.5-7b-L23"


def test_warrant_round_trips_including_a_refusal(ledger: Ledger) -> None:
    """Refusals are stored exactly like issuances.

    A log recording only the warrants that were granted would let a refusal be
    retried quietly until it passed.
    """
    refused = make_warrant(
        eval_set_id="triviaqa-longctx-600",
        controls=failing_controls("null_feature"),
        status=WarrantStatus.REFUSED,
        status_reason="null_feature scored 0.71, outside [0.45, 0.55]",
    )
    ledger.append_warrant(refused)
    restored = ledger.get_warrant(refused.warrant_id)
    assert restored == refused
    assert restored.status is WarrantStatus.REFUSED
    assert "0.71" in restored.status_reason


def test_appending_the_same_record_twice_is_refused(ledger: Ledger) -> None:
    """A changed record is a new record, never an overwrite of an old one."""
    warrant = make_warrant()
    ledger.append_warrant(warrant)
    with pytest.raises(LedgerError, match="already in the ledger"):
        ledger.append_warrant(warrant)


def test_a_sealed_certificate_cannot_be_appended_again(ledger: Ledger) -> None:
    sealed = ledger.append_certificate(make_certificate())
    with pytest.raises(LedgerError, match="already sealed"):
        ledger.append_certificate(sealed)


# --------------------------------------------------------------------------- #
# The warrant matrix's read path
# --------------------------------------------------------------------------- #


def test_missing_cell_reads_as_absence_not_as_a_record(ledger: Ledger) -> None:
    """None is UNVALIDATED (``DECISIONS.md`` 024), and cannot be dereferenced."""
    key = WarrantKey("probe-qwen2.5-7b-L23", "P-conservative", "hinglish-pii-200")
    assert ledger.latest_warrant(key) is None
    assert ledger.warrants_for_key(key) == ()


def test_revalidation_appends_rather_than_replaces(ledger: Ledger) -> None:
    """The history of a cell stays readable, refusals included."""
    first = make_warrant(validation_run_id="run-0001")
    ledger.append_warrant(first)
    second = make_warrant(
        validation_run_id="run-0002",
        controls=failing_controls("canary"),
        status=WarrantStatus.REFUSED,
        status_reason="canary recall 0.85, below 1.0",
    )
    ledger.append_warrant(second)

    history = ledger.warrants_for_key(first.key)
    assert len(history) == 2
    assert [w.validation_run_id for w in history] == ["run-0001", "run-0002"]
    assert ledger.latest_warrant(first.key).status is WarrantStatus.REFUSED


def test_two_envelopes_are_two_cells(ledger: Ledger) -> None:
    """Invariant 1 at the storage layer, not only in the type."""
    short = make_warrant(eval_set_id="triviaqa-600")
    long = make_warrant(eval_set_id="triviaqa-longctx-600")
    ledger.append_warrant(short)
    ledger.append_warrant(long)

    assert ledger.latest_warrant(short.key).eval_set_id == "triviaqa-600"
    assert ledger.latest_warrant(long.key).eval_set_id == "triviaqa-longctx-600"
    assert ledger.count(RecordKind.WARRANT) == 2


# --------------------------------------------------------------------------- #
# DPDP Rule 6 query paths
# --------------------------------------------------------------------------- #


def test_query_by_session_time_policy_detector_status_and_category(
    ledger: Ledger,
) -> None:
    """Every predicate DPDP Rule 6 names, and they compose."""
    warrant = make_warrant()
    ledger.append_warrant(warrant)
    start = utc_now()

    ledger.append_certificate(
        make_certificate(
            certificate_id="C-pii", request_id="R-1", session_id="S-alpha",
            warrant=warrant,
            findings=(make_finding("F-1", category=Category.PII,
                                   warrant_id=warrant.warrant_id),),
        )
    )
    ledger.append_certificate(
        make_certificate(
            certificate_id="C-halluc", request_id="R-2", session_id="S-beta",
            warrant=warrant,
        )
    )

    by_session = ledger.certificates_for_session("S-alpha")
    assert [c.certificate_id for c in by_session] == ["C-pii"]

    assert len(ledger.query(kind=RecordKind.CERTIFICATE, category="PII")) == 1
    assert len(ledger.query(kind=RecordKind.CERTIFICATE, policy_version="3.1")) == 2
    assert len(ledger.query(kind=RecordKind.CERTIFICATE, policy_version="9.9")) == 0
    assert len(ledger.query(detector_version="probe-qwen2.5-7b-L23@1.0.0+ab12cd34")) == 3
    assert len(ledger.query(kind=RecordKind.WARRANT,
                            warrant_status=WarrantStatus.VALID)) == 1
    assert len(ledger.query(since=start - timedelta(minutes=1))) == 3
    assert len(ledger.query(since=start + timedelta(days=1))) == 0

    # Conjunctions, which is how the questions actually arrive.
    combined = ledger.query(
        kind=RecordKind.CERTIFICATE, session_id="S-alpha", category="PII",
        policy_version="3.1", since=start - timedelta(minutes=1),
    )
    assert [r.record_id for r in combined] == ["C-pii"]


def test_query_boundaries_must_be_timezone_aware(ledger: Ledger) -> None:
    from datetime import datetime as _dt

    with pytest.raises(LedgerError, match="naive"):
        ledger.query(since=_dt(2026, 1, 1))


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #


def test_retention_below_one_year_is_refused(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="at least 365"):
        Ledger(tmp_path / "short.db", retention_days=90)


def test_purge_refuses_to_breach_the_retention_floor(ledger: Ledger) -> None:
    """DPDP Rule 6 sets a minimum, and this store will not go under it."""
    ledger.append_warrant(make_warrant())
    with pytest.raises(LedgerError, match="retention floor"):
        ledger.purge_older_than(utc_now() - timedelta(days=30), dry_run=False)


def test_purge_is_a_dry_run_by_default(ledger: Ledger) -> None:
    ledger.append_warrant(make_warrant())
    now = utc_now() + timedelta(days=500)
    summary = ledger.purge_older_than(now - timedelta(days=450), now=now)
    assert summary["dry_run"] is True
    assert summary["n_records"] == 1
    assert ledger.count() == 1


def test_a_declared_purge_leaves_the_chain_verifiable(tmp_path: Path) -> None:
    """The log stays honest about having been shortened.

    Deleting from a hash chain breaks it. Pretending otherwise would be worse
    than not purging: the chain would verify over a log that no longer contains
    what it claims to. So the purge is itself a record naming the hash it
    removed up to, and verification checks the surviving anchor against it.
    """
    clock = _Clock(utc_now())
    store = Ledger(tmp_path / "aged.db", retention_days=400, clock=clock)
    warrant = make_warrant()
    store.append_warrant(warrant)
    store.append_certificate(make_certificate(certificate_id="C-old", request_id="R-0",
                                              warrant=warrant))
    clock.advance(timedelta(days=500))
    store.append_certificate(make_certificate(certificate_id="C-new", request_id="R-1",
                                              warrant=warrant))
    assert store.verify_chain().ok

    summary = store.purge_older_than(clock.now - timedelta(days=400), dry_run=False)
    assert summary["n_records"] == 2
    assert summary["dry_run"] is False

    after = store.verify_chain()
    assert after.is_truncated
    assert after.ok, after.breaks
    assert store.count() == 2  # the surviving certificate, plus the purge record

    events = store.query(kind=RecordKind.RETENTION_EVENT)
    assert len(events) == 1
    assert events[0].body["last_hash_removed"] == after.anchor_hash
    # The surviving records are still readable and still typed.
    assert store.get_certificate("C-new").request_id == "R-1"
    with pytest.raises(LedgerError, match="no certificate"):
        store.get_certificate("C-old")
    store.close()


def test_total_erasure_is_undetectable_and_we_say_so(tmp_path: Path) -> None:
    """The honest limit of a self-contained hash chain.

    An attacker who can delete every row can also delete the retention event
    that would have declared the deletion, and an empty ledger verifies. No log
    that is its own only witness can do better; detecting total erasure needs an
    anchor outside the file. This test exists so the limitation is asserted
    rather than discovered, and so it cannot be quietly claimed away later.
    """
    store = Ledger(tmp_path / "erased.db", retention_days=400)
    store.append_warrant(make_warrant())
    store.append_certificate(make_certificate())
    assert store.verify_chain().ok

    conn = sqlite3.connect(store.path)
    conn.execute("DELETE FROM ledger")
    conn.commit()
    conn.close()

    erased = store.verify_chain()
    assert erased.ok and erased.n_records == 0
    assert not erased.is_truncated
    store.close()


class _Clock:
    """A movable clock, so retention can be tested without waiting a year."""

    def __init__(self, start) -> None:
        self.now = start

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta

    def __call__(self):
        return self.now
