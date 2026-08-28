"""The Presidio adapter. ``DECISIONS.md`` 084.

Presidio pulls spaCy and a language model, so every test here skips when it is
absent — and the skip is loud rather than a silent pass, because an adapter that
does not run is how a detector's absence gets mistaken for a clean sheet.

The tests are weighted towards the claims the Phase 8 argument rests on: that
"stock" really is stock, that the eval-set identifiers are not the confound, and
that the adapter does not quietly misrepresent the tool in either direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detectors.presidio_adapter import (
    CONFIGURATIONS,
    INDIAN_RECOGNIZERS,
    PresidioDetector,
    presidio_available,
)

pytestmark = pytest.mark.skipif(
    not presidio_available(),
    reason="presidio-analyzer not installed; `pip install presidio-analyzer`",
)

ROOT = Path(__file__).resolve().parents[1]


def evalset(name: str) -> dict:
    return json.loads((ROOT / "evalsets" / f"{name}.json").read_text(encoding="utf-8"))


def test_stock_registers_no_indian_recognizer() -> None:
    """The premise of the whole Phase 8 argument, checked rather than asserted.

    If Presidio ever starts registering these by default, the "day one" claim
    stops being true and this test is where that surfaces.
    """
    registry = PresidioDetector("stock").analyzer.registry
    names = {r.name for r in registry.recognizers}
    assert names, "a stock engine should still load its non-Indian recognizers"
    assert not (names & set(INDIAN_RECOGNIZERS)), (
        f"stock Presidio registered {sorted(names & set(INDIAN_RECOGNIZERS))}"
    )


def test_enabling_adds_exactly_the_shipped_indian_recognizers() -> None:
    names = {r.name for r in PresidioDetector("enabled").analyzer.registry.recognizers}
    assert set(INDIAN_RECOGNIZERS) <= names


def test_the_configuration_is_part_of_the_detector_identity() -> None:
    """Two configurations have different measured bounds, so they are different
    detectors as far as the warrant key is concerned. A shared id would let a
    warrant measured on one be quoted for the other."""
    ids = {PresidioDetector(c).detector_id for c in CONFIGURATIONS}
    assert len(ids) == len(CONFIGURATIONS)


def test_an_unknown_configuration_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown Presidio configuration"):
        PresidioDetector("enabled_plus_tuning")


def test_the_aadhaar_validator_runs_verhoeff_not_just_a_pattern() -> None:
    """``TASKS.md`` Phase 8 asks what ``InAadhaarRecognizer`` actually validates.

    A twelve-digit number that fails the Verhoeff check must be rejected; the
    same number corrected must pass. If this ever becomes pattern-only, every
    recall number in 084 changes meaning.
    """
    from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer

    recognizer = InAadhaarRecognizer()
    valid = "999906872026"
    assert recognizer.validate_result(valid)
    # Same length, same leading digits, broken check digit.
    assert not recognizer.validate_result(valid[:-1] + ("7" if valid[-1] != "7" else "8"))
    # Leading digit below 2 is not a real Aadhaar.
    assert not recognizer.validate_result("199906872026")


def test_the_builtin_sanitiser_does_not_strip_dots() -> None:
    """The measured cause of the 'spaced' miss, pinned.

    The fix in ``presidio_custom`` widens the pattern and the separator set; if
    upstream ever strips dots itself, that custom work becomes redundant and
    this test says so.
    """
    from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer

    recognizer = InAadhaarRecognizer()
    assert recognizer.validate_result("9999 0687 2026")
    assert not recognizer.validate_result("99.99.06.87.20.26")


def test_the_fixture_identifiers_are_not_the_confound() -> None:
    """A low recall must not be made-up numbers correctly failing a checksum.

    ``hinglish-pii-200`` records ``checksum_valid`` per item; this asserts the
    flag agrees with Verhoeff on every Aadhaar item, which is what licenses
    reading recall as a property of the detector.
    """
    from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer

    recognizer = InAadhaarRecognizer()
    checked = 0
    for item in evalset("hinglish-pii-200")["items"]:
        meta = item.get("meta") or {}
        if meta.get("identifier_kind") != "IN_AADHAAR":
            continue
        checked += 1
        assert recognizer.validate_result(str(meta["canonical"])) == meta["checksum_valid"]
    assert checked >= 30, f"only {checked} Aadhaar items found; the set changed"


def test_the_allowlist_covers_every_entity_the_custom_recognizers_emit() -> None:
    """The bug 084 records: an allowlist that drops a detector's own output.

    ``UPI_VPA`` and ``IN_IFSC`` are emitted only by the custom recognizers and
    were missing from the allowlist, which understated ``enabled_plus_custom``
    and refused a warrant it should have been issued.
    """
    from src.detectors.presidio_adapter import _IDENTIFIER_ENTITIES
    from src.detectors.presidio_custom import custom_recognizers

    emitted = set()
    for recognizer in custom_recognizers():
        emitted.update(recognizer.supported_entities)
    missing = sorted(emitted - _IDENTIFIER_ENTITIES)
    assert not missing, (
        f"the adapter filters out {missing}, which its own custom recognizers "
        "emit; that understates the configuration those recognizers exist for"
    )


def test_more_recognizers_never_lowers_canary_recall() -> None:
    """Monotonicity across the three configurations on the easy set.

    Not a claim that enabling always helps in general — a looser pattern can
    cost precision — but on twenty verbatim checksum-valid identifiers, adding
    recognizers must not lose one. If it does, a custom recognizer is shadowing
    a built-in.
    """
    canary = evalset("canary-20-pii")
    texts = [item["prompt"] for item in canary["items"]]
    caught = []
    for configuration in CONFIGURATIONS:
        detector = PresidioDetector(configuration)
        scores = detector.score(texts)
        caught.append(int((scores >= detector.min_confidence).sum()))
    assert caught == sorted(caught), f"canary recall is not monotone: {caught}"
    assert caught[0] < caught[-1], "the configurations are indistinguishable"


def test_the_score_is_a_maximum_not_a_count() -> None:
    """The label is 'this message contains an identifier', so two identifiers is
    still one positive. Summing would make verbose disclosures score higher for
    no reason connected to the label."""
    detector = PresidioDetector("enabled")
    one = "aadhaar number 9999 0687 2026 hai"
    two = one + " aur dusra 9999 7630 4615 bhi"
    scores = detector.score([one, two])
    assert scores[1] == pytest.approx(scores[0])
    assert 0.0 <= scores.min() and scores.max() <= 1.0


def test_a_message_with_no_identifier_scores_zero() -> None:
    detector = PresidioDetector("enabled")
    assert detector.score(["Account balance kitna hai bhai?"])[0] == 0.0


# --------------------------------------------------------------------------- #
# The allowlist is a hard failure, not a filter — DECISIONS.md 086
# --------------------------------------------------------------------------- #


def test_an_unclassified_entity_raises_rather_than_being_dropped() -> None:
    """The general form of the bug 084 hit, closed for every future adapter.

    A filter lets any adapter understate its detector with nothing in the output
    to show for it, and the understatement is in the direction nobody audits.
    Every entity must be classified as an identifier or as ignorable, so the
    lists have to be extended deliberately.
    """
    from src.detectors import presidio_adapter
    from src.detectors.presidio_adapter import UnclassifiedEntityError

    detector = PresidioDetector("stock")
    original = presidio_adapter._IGNORED_ENTITIES
    try:
        # Remove a class the analyzer really emits on this input.
        presidio_adapter._IGNORED_ENTITIES = frozenset(original - {"DATE_TIME", "URL"})
        with pytest.raises(UnclassifiedEntityError, match="classifies neither"):
            detector.find("Call me on 12 January 2026 or see https://example.com")
    finally:
        presidio_adapter._IGNORED_ENTITIES = original


def test_every_entity_the_shipped_sets_provoke_is_classified() -> None:
    """The guard must not fire on our own corpora — otherwise it is a landmine
    rather than a check."""
    from src.detectors.presidio_adapter import UnclassifiedEntityError

    for name in ("hinglish-pii-200", "hinglish-pii-200b", "hard-negatives-200", "canary-20-pii"):
        texts = [item["prompt"] for item in evalset(name)["items"]]
        for configuration in CONFIGURATIONS:
            try:
                PresidioDetector(configuration).score(texts)
            except UnclassifiedEntityError as exc:  # pragma: no cover - failure path
                pytest.fail(f"{configuration} on {name}: {exc}")


def test_the_holdout_is_a_different_envelope_with_the_same_shape() -> None:
    """``hinglish-pii-200b`` must be a new identity, not a renamed copy."""
    base, holdout = evalset("hinglish-pii-200"), evalset("hinglish-pii-200b")
    assert base["eval_set_id"] != holdout["eval_set_id"]
    assert base["content_hash"] != holdout["content_hash"]
    assert len(base["items"]) == len(holdout["items"])
    prompts = {i["prompt"] for i in base["items"]}
    overlap = sum(1 for i in holdout["items"] if i["prompt"] in prompts)
    assert overlap < len(holdout["items"]) // 4, (
        f"{overlap} of {len(holdout['items'])} holdout prompts are identical to "
        "the base set; the identifiers were meant to be redrawn"
    )


def test_the_extended_inventory_does_not_change_the_frozen_set() -> None:
    """Construction notes are inside the content hash.

    Recording ``extended_forms`` unconditionally would change the identity of
    ``hinglish-pii-200`` and orphan every warrant keyed on it, so the key is
    written only when the extension is used.
    """
    from src.evalsets.builders import build_hinglish_pii

    rebuilt = build_hinglish_pii(seed=1729)
    assert rebuilt.content_hash == evalset("hinglish-pii-200")["content_hash"]
    assert "extended_forms" not in rebuilt.construction
