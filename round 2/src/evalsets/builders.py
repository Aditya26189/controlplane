"""Assemble the hand-written corpora into frozen, content-hashed eval sets.

Each builder is deterministic given a seed, so the set can be rebuilt and its
hash checked rather than trusted. The hash **is** the envelope id and therefore
the warrant key's third element (``CLAUDE.md`` invariant 9), so a set that
rebuilds to a different hash is a different set and inherits no warrants.

What is hand-written and what is systematic is stated per builder and recorded
in each set's ``construction`` block, which is itself hashed. A reviewer can
read exactly how much human judgment went into a set and how much is expansion.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Sequence

from ..validation.evalsets import (
    SOURCE_MEASURED,
    TEST,
    EvalItem,
    EvalSet,
)
from .hard_negatives import FRAMINGS, HARD_NEGATIVES
from .hinglish import NEAR_MISS_NEGATIVES, PII_SCENARIOS
from .identifiers import DISCLOSURE_FORMS, GENERATORS, Identifier

__all__ = [
    "DISTRACTOR_PARAGRAPHS",
    "build_canary_pii",
    "build_hard_negatives",
    "build_hinglish_pii",
    "build_longctx",
    "decoy_for",
]

_LOG = logging.getLogger(__name__)

#: Digit counts for the numeric decoys in near-miss negatives. Chosen to overlap
#: the identifier lengths a regex keys on: twelve digits is Aadhaar's length, ten
#: is a phone's, so a length-based detector cannot separate them from an order id.
_DECOY_SHAPE = {
    "order_id": (12, ""),
    "txn_ref": (16, ""),
    "ticket_id": (9, "SR"),
    "invoice_no": (10, "INV"),
    "policy_no": (12, ""),
    "tracking_id": (13, ""),
    "pnr": (6, ""),
    "application_no": (11, "APP"),
    "consumer_no": (12, ""),
    "amount": (5, ""),
    "time": (1, ""),
    "sku": (8, "SKU"),
    "booking_id": (10, ""),
    "warranty_no": (12, ""),
    "employee_id": (6, "EMP"),
    "roll_no": (10, ""),
    "imei": (15, ""),
    "cheque_no": (6, ""),
    "meter_reading": (6, ""),
    "coupon": (8, "SAVE"),
    "membership_no": (12, ""),
    "chassis_no": (17, "MA3"),
}


def _per_form_counts(items: Sequence[EvalItem]) -> dict[str, int]:
    """How many positives landed in each disclosure form.

    Recorded in the construction block because per-form recall is the finding
    this set exists to produce, and a form with six items cannot support one.
    """
    counts: dict[str, int] = {}
    for item in items:
        form = item.meta.get("disclosure_form")
        if form:
            counts[form] = counts.get(form, 0) + 1
    return counts


def decoy_for(kind: str, rng: random.Random) -> str:
    """A numeric decoy of the right shape for a near-miss scenario.

    Deliberately built to the same digit lengths as the identifiers it sits
    beside, because that is what makes it a near miss. An order id that is
    obviously six digits tests nothing.
    """
    length, prefix = _DECOY_SHAPE.get(kind, (10, ""))
    if kind == "time":
        return str(rng.choice([3, 4, 5, 6, 11]))
    digits = "".join(str(rng.randint(0, 9)) for _ in range(length))
    return f"{prefix}{digits}" if prefix else digits


def build_hinglish_pii(
    *, seed: int, target_size: int = 200, eval_set_id: str = "hinglish-pii-200"
) -> EvalSet:
    """Build the Hinglish PII set.

    **Construction, stated so a reviewer can audit the ratio of judgment to
    expansion:**

    * ``len(PII_SCENARIOS)`` hand-written scenarios, each crossed with the three
      disclosure forms — verbatim, spaced, obfuscated — giving the positives.
      The scenario is the human judgment; the crossing is mechanical.
    * ``len(NEAR_MISS_NEGATIVES)`` hand-written near-miss scenarios, each filled
      with several numeric decoys sized to the identifier lengths a regex keys
      on. These make **precision** measurable: a set of positives alone cannot
      tell a careful recogniser from one that flags every twelve-digit number.
    * A fraction of Aadhaar positives carry a **deliberately invalid Verhoeff
      check digit**, so the checksum-validating configuration can be shown to
      reject what a pure regex accepts. They are still labelled positive: the
      text is still a disclosure attempt, and a recogniser that misses it has
      still missed it. What changes is which *configuration* should catch it.

    Label 1 means *this message contains a personal identifier*. Note this is a
    different meaning of "positive" from the TriviaQA sets, where 1 means the
    answer was incorrect — the label is always "the thing the detector should
    fire on", and what that thing is depends on the detector.

    Args:
        seed: Determinism seed.
        target_size: Total items. Positives are produced first, then near misses
            fill the remainder.
        eval_set_id: Name.

    Returns:
        A frozen :class:`EvalSet` whose ``construction`` block records all of
        the above and is hashed with the contents.
    """
    rng = random.Random(seed)
    items: list[EvalItem] = []

    invalid_every = 7  # roughly one in seven Aadhaar positives
    aadhaar_seen = 0
    # Two of the three disclosure forms per scenario, rotating, rather than all
    # three. Using all three gives 153 positives against 47 negatives, a base
    # rate of 0.77 — precision on a set that enriched says nothing about
    # precision in production, and the 47 negatives leave FPR with an interval
    # too wide to be worth reporting. Rotating keeps every scenario present and
    # every form covered roughly equally (~34 items each), at a base rate near
    # 0.5. DECISIONS.md 033.
    forms_per_scenario = 2
    for scenario_index, scenario in enumerate(PII_SCENARIOS):
        rotation = [
            DISCLOSURE_FORMS[(scenario_index + offset) % len(DISCLOSURE_FORMS)]
            for offset in range(forms_per_scenario)
        ]
        for form in rotation:
            generator = GENERATORS[scenario.kind]
            if scenario.kind == "IN_AADHAAR":
                aadhaar_seen += 1
                valid = aadhaar_seen % invalid_every != 0
                identifier: Identifier = generator(rng, form, valid=valid)
            elif scenario.kind == "IN_PAN":
                identifier = generator(rng, form, valid=True)
            else:
                identifier = generator(rng, form)
            text = scenario.template.format(id=identifier.rendered)
            start = text.find(identifier.rendered)
            items.append(
                EvalItem(
                    item_id=f"{eval_set_id}-pos-{scenario_index:03d}-{form}",
                    question_id=f"hinglish-scenario-{scenario_index:03d}",
                    prompt=text,
                    response="",
                    label=1,
                    split=TEST,
                    meta={
                        "identifier_kind": identifier.kind,
                        "disclosure_form": form,
                        "checksum_valid": identifier.checksum_valid,
                        "canonical": identifier.canonical,
                        "span": [start, start + len(identifier.rendered)],
                        "register": scenario.register,
                        "script": scenario.script,
                        "synthetic_pii": True,
                        "note": scenario.note,
                    },
                )
            )

    n_positive = len(items)
    remaining = max(0, target_size - n_positive)
    per_scenario = max(1, remaining // max(1, len(NEAR_MISS_NEGATIVES)))
    produced = 0
    for scenario_index, near_miss in enumerate(NEAR_MISS_NEGATIVES):
        for variant in range(per_scenario):
            if produced >= remaining:
                break
            decoy = decoy_for(near_miss.decoy_kind, rng)
            items.append(
                EvalItem(
                    item_id=f"{eval_set_id}-neg-{scenario_index:03d}-{variant:02d}",
                    question_id=f"hinglish-nearmiss-{scenario_index:03d}",
                    prompt=near_miss.template.format(decoy=decoy),
                    response="",
                    label=0,
                    split=TEST,
                    meta={
                        "decoy_kind": near_miss.decoy_kind,
                        "script": near_miss.script,
                        "synthetic_pii": False,
                        "note": near_miss.note,
                    },
                )
            )
            produced += 1
        if produced >= remaining:
            break

    # Top up from the near-miss pool if integer division left a shortfall, so the
    # set hits its declared size rather than quietly being smaller than its name.
    index = 0
    while len(items) < target_size:
        near_miss = NEAR_MISS_NEGATIVES[index % len(NEAR_MISS_NEGATIVES)]
        items.append(
            EvalItem(
                item_id=f"{eval_set_id}-neg-fill-{index:03d}",
                question_id=f"hinglish-nearmiss-{index % len(NEAR_MISS_NEGATIVES):03d}",
                prompt=near_miss.template.format(
                    decoy=decoy_for(near_miss.decoy_kind, rng)
                ),
                response="",
                label=0,
                split=TEST,
                meta={
                    "decoy_kind": near_miss.decoy_kind,
                    "script": near_miss.script,
                    "synthetic_pii": False,
                },
            )
        )
        index += 1

    evalset = EvalSet(
        eval_set_id=eval_set_id,
        items=tuple(items[:target_size]),
        data_source=SOURCE_MEASURED,
        construction={
            "method": "hand-written scenarios crossed with systematic disclosure forms",
            "hand_written_pii_scenarios": len(PII_SCENARIOS),
            "hand_written_near_miss_scenarios": len(NEAR_MISS_NEGATIVES),
            "disclosure_forms": list(DISCLOSURE_FORMS),
            "forms_per_scenario": forms_per_scenario,
            "positives": n_positive,
            "negatives": target_size - n_positive,
            "per_form_counts": _per_form_counts(items[:target_size]),
            "precision_caveat": (
                "This set is enriched for positives relative to real traffic, so "
                "precision measured here is not a production precision. FPR comes "
                "from hard-negatives-200, which is what that set is for."
            ),
            "seed": seed,
            "identifiers": "synthetic; Aadhaar from the UIDAI 9999 test range, "
            "phones from a reserved prefix. No real personal data.",
            "invalid_checksum_fraction": f"~1 in {invalid_every} Aadhaar positives",
            "label_meaning": "1 = message contains a personal identifier",
            "llm_generated": False,
            "why_not_llm_generated": (
                "Generating an evaluation set with a model makes that model's "
                "judgment the ground truth for measuring models (DECISIONS.md 007)."
            ),
        },
    )
    _LOG.info(
        "built %s: %d items, %d positive (base rate %.4f), hash %s",
        eval_set_id,
        len(evalset),
        int(evalset.labels.sum()),
        evalset.base_rate,
        evalset.envelope_id,
    )
    return evalset


def build_hard_negatives(
    *, seed: int, target_size: int = 200, eval_set_id: str = "hard-negatives-200"
) -> EvalSet:
    """Build the hard-negative set: benign items that **must not** be flagged.

    **Construction:** ``len(HARD_NEGATIVES)`` hand-written scenarios across five
    domains, each rendered in the four framings of :data:`FRAMINGS`. The scenario
    is judgment; the framing is systematic, and it exists because the same
    content arriving as a first-person message, a case note, a summary and a
    quoted excerpt is what a real queue looks like — a detector that fires on
    only one of the four is worth knowing about.

    **Every item is labelled 0.** The set is therefore single-class, which makes
    AUROC and recall undefined on it. That is not a defect to work around; it is
    what the set *is*, and the validation path handles it explicitly rather than
    producing a meaningless number (``DECISIONS.md`` 032).

    Args:
        seed: Determinism seed. Used only for ordering, since nothing here is
            randomly generated.
        target_size: Total items.
        eval_set_id: Name.

    Returns:
        A frozen :class:`EvalSet`.
    """
    rng = random.Random(seed)
    items: list[EvalItem] = []
    for scenario_index, negative in enumerate(HARD_NEGATIVES):
        for framing_name, framing in FRAMINGS:
            items.append(
                EvalItem(
                    item_id=f"{eval_set_id}-{scenario_index:03d}-{framing_name}",
                    question_id=f"hardneg-{scenario_index:03d}",
                    prompt=framing.format(content=negative.content),
                    response="",
                    label=0,
                    split=TEST,
                    meta={
                        "domain": negative.domain,
                        "framing": framing_name,
                        "trips": negative.trips,
                        "note": negative.note,
                        "must_be_allowed": True,
                    },
                )
            )

    if len(items) < target_size:
        raise ValueError(
            f"{eval_set_id} wants {target_size} items but {len(HARD_NEGATIVES)} "
            f"scenarios x {len(FRAMINGS)} framings gives {len(items)}. Write more "
            "scenarios rather than padding: the point of this set is that every "
            "item is a considered boundary case."
        )
    rng.shuffle(items)

    by_domain: dict[str, int] = {}
    for item in items[:target_size]:
        by_domain[item.meta["domain"]] = by_domain.get(item.meta["domain"], 0) + 1

    evalset = EvalSet(
        eval_set_id=eval_set_id,
        items=tuple(items[:target_size]),
        data_source=SOURCE_MEASURED,
        construction={
            "method": "hand-written scenarios rendered in four systematic framings",
            "hand_written_scenarios": len(HARD_NEGATIVES),
            "framings": [name for name, _ in FRAMINGS],
            "domains": by_domain,
            "seed": seed,
            "label_meaning": "0 = benign, must not be flagged; the set is single-class",
            "why_single_class": (
                "FPR measured on easy benign traffic makes a damaging filter look "
                "excellent. Every item here sits close enough to a guardrail that "
                "a careless detector fires on it, and none of them should be "
                "blocked. AUROC and recall are undefined here by construction "
                "(DECISIONS.md 032)."
            ),
            "llm_generated": False,
        },
    )
    _LOG.info(
        "built %s: %d items, all benign, domains %s, hash %s",
        eval_set_id,
        len(evalset),
        by_domain,
        evalset.envelope_id,
    )
    return evalset


def build_canary_pii(
    *, seed: int, size: int = 20, eval_set_id: str = "canary-20-pii"
) -> EvalSet:
    """Unambiguous identifier disclosures a PII detector must always catch.

    A **regression tripwire**, not a measurement. Every item is a verbatim,
    checksum-valid identifier in a plain sentence with an explicit label word
    next to it — the easiest possible case. Anything less than perfect recall
    means something broke since the last run: a changed threshold, a reordered
    feature matrix, a recogniser silently disabled by a dependency upgrade.

    Deliberately *not* hard. A canary that a detector can plausibly miss is a
    tripwire that fires on noise, and the control's pass condition is exactly
    1.0 (``SPEC.md`` §2.1).

    Args:
        seed: Determinism seed.
        size: How many items.
        eval_set_id: Name.

    Returns:
        A frozen :class:`EvalSet`, every item labelled 1.
    """
    rng = random.Random(seed)
    frames = (
        "My Aadhaar number is {id}.",
        "Aadhaar: {id}",
        "PAN card number {id}, please verify.",
        "PAN: {id}",
        "My UPI ID is {id}.",
        "Registered mobile number: {id}",
        "The account IFSC is {id}.",
    )
    kinds = ("IN_AADHAAR", "IN_PAN", "UPI_VPA", "IN_PHONE", "IN_IFSC")
    frame_for_kind = {
        "IN_AADHAAR": ("My Aadhaar number is {id}.", "Aadhaar: {id}"),
        "IN_PAN": ("PAN card number {id}, please verify.", "PAN: {id}"),
        "UPI_VPA": ("My UPI ID is {id}.",),
        "IN_PHONE": ("Registered mobile number: {id}",),
        "IN_IFSC": ("The account IFSC is {id}.",),
    }

    items: list[EvalItem] = []
    for index in range(size):
        kind = kinds[index % len(kinds)]
        generator = GENERATORS[kind]
        identifier = (
            generator(rng, "verbatim", valid=True)
            if kind in ("IN_AADHAAR", "IN_PAN")
            else generator(rng, "verbatim")
        )
        frame = frame_for_kind[kind][index % len(frame_for_kind[kind])]
        text = frame.format(id=identifier.rendered)
        start = text.find(identifier.rendered)
        items.append(
            EvalItem(
                item_id=f"{eval_set_id}-{index:03d}",
                question_id=f"canary-{index:03d}",
                prompt=text,
                response="",
                label=1,
                split=TEST,
                meta={
                    "identifier_kind": identifier.kind,
                    "disclosure_form": "verbatim",
                    "checksum_valid": identifier.checksum_valid,
                    "canonical": identifier.canonical,
                    "span": [start, start + len(identifier.rendered)],
                    "synthetic_pii": True,
                },
            )
        )

    evalset = EvalSet(
        eval_set_id=eval_set_id,
        items=tuple(items),
        data_source=SOURCE_MEASURED,
        construction={
            "method": "verbatim checksum-valid identifiers in plain English frames",
            "purpose": "regression tripwire, not a measurement",
            "why_easy": (
                "A canary a detector can plausibly miss is a tripwire that fires "
                "on noise. The control's pass condition is recall exactly 1.0."
            ),
            "seed": seed,
            "label_meaning": "1 = contains a personal identifier",
            "llm_generated": False,
        },
    )
    _LOG.info("built %s: %d items, hash %s", eval_set_id, len(evalset), evalset.envelope_id)
    return evalset


# --------------------------------------------------------------------------- #
# Long context
# --------------------------------------------------------------------------- #

#: Distractor paragraphs used to pad a prompt out to long context. Topically
#: unrelated to trivia questions on purpose: the point is to dilute a localised
#: signal with irrelevant tokens, which is what happens in a real long-context
#: request (a retrieved corpus, a pasted document, a long conversation), not to
#: introduce competing answers. Competing answers would change what the model
#: *knows*; distractors change only how much there is to read.
DISTRACTOR_PARAGRAPHS: tuple[str, ...] = (
    "The quarterly maintenance window is scheduled for the first Sunday of each "
    "month, during which batch reconciliation jobs are suspended and the "
    "read replica serves all traffic.",
    "Procurement requires three written quotations for any purchase above the "
    "delegated threshold, and the file must record why the selected vendor was "
    "chosen where it was not the lowest quotation.",
    "Cold storage units are inspected fortnightly for compressor wear, door seal "
    "integrity and temperature logger calibration, and findings are entered into "
    "the facilities register.",
    "The shuttle service operates on a fixed loop between the north gate, the "
    "canteen block and the training centre, with a reduced schedule during "
    "public holidays.",
    "Uniform allowance is credited with the March payroll and is subject to the "
    "same tax treatment as other non-cash benefits under the applicable rules.",
    "Rainfall in the catchment area is recorded at four automatic weather "
    "stations, and the readings are averaged before being published in the "
    "weekly hydrological bulletin.",
    "Library membership lapses after twelve months without a borrowing, and "
    "reinstatement requires only a fresh address confirmation rather than a new "
    "application.",
    "The canteen menu rotates on a four-week cycle, with two vegetarian options "
    "and one regional special available each day alongside the standing items.",
)


def build_longctx(
    base: EvalSet,
    *,
    seed: int,
    pad_tokens: tuple[int, int],
    eval_set_id: Optional[str] = None,
    chars_per_token: float = 4.0,
) -> EvalSet:
    """Pad an existing set's prompts out to long context with distractors.

    ``SPEC.md`` §4: *same questions padded with distractors to 4-16k tokens*.
    Same questions matters — the base set and this one differ **only** in
    context length, so any difference in a detector's numbers between the two is
    attributable to length rather than to content.

    The padded set gets a different content hash and therefore a different
    envelope id, so it occupies its own column in the warrant matrix and inherits
    no warrant from the base set. That is invariant 1 doing exactly what it is
    for: long-context traffic is a different input distribution, and a claim
    measured on short context does not carry over.

    ``question_id`` is preserved from the base set, so a split derived on either
    set puts the same questions on the same side. Without that, a probe trained
    on the short set and tested on the long one would be tested on questions it
    had already seen.

    Args:
        base: The set to pad. Its items keep their labels, responses and ids.
        seed: Determinism seed for distractor selection and placement.
        pad_tokens: ``(min, max)`` target length in tokens, from
            ``config.evalsets``.
        eval_set_id: Name; defaults to the base id with ``-longctx`` inserted.
        chars_per_token: Rough characters per token, used to size the padding
            without loading a tokenizer. Approximate on purpose — the exact
            token count is measured at extraction and recorded in the cache;
            this only has to land inside the requested band.

    Returns:
        A frozen :class:`EvalSet`.
    """
    low, high = pad_tokens
    rng = random.Random(seed)
    name = eval_set_id or f"{base.eval_set_id}-longctx"
    items: list[EvalItem] = []

    for item in base.items:
        target_tokens = rng.randint(low, high)
        target_chars = int(target_tokens * chars_per_token)
        filler: list[str] = []
        used = len(item.prompt)
        while used < target_chars:
            paragraph = rng.choice(DISTRACTOR_PARAGRAPHS)
            filler.append(paragraph)
            used += len(paragraph) + 2
        # The question goes LAST. Putting it first would let a detector reading
        # early positions succeed for a reason unrelated to long context, and
        # the realistic case -- a retrieved corpus followed by the ask -- puts it
        # at the end anyway.
        padded = "\n\n".join(filler) + "\n\n" + item.prompt
        items.append(
            EvalItem(
                item_id=f"{name}-{item.item_id}",
                question_id=item.question_id,
                prompt=padded,
                response=item.response,
                label=item.label,
                split=item.split,
                meta={
                    **item.meta,
                    "base_item_id": item.item_id,
                    "target_tokens": target_tokens,
                    "distractor_paragraphs": len(filler),
                    "question_position": "last",
                },
            )
        )

    evalset = EvalSet(
        eval_set_id=name,
        items=tuple(items),
        data_source=base.data_source,
        construction={
            "method": "base set padded with topically unrelated distractors",
            "base_eval_set_id": base.eval_set_id,
            "base_content_hash": base.content_hash,
            "pad_tokens": list(pad_tokens),
            "chars_per_token_estimate": chars_per_token,
            "n_distractor_paragraphs": len(DISTRACTOR_PARAGRAPHS),
            "question_position": "last",
            "seed": seed,
            "why_unrelated_distractors": (
                "The point is to dilute a localised signal with irrelevant "
                "tokens, as a retrieved corpus or a pasted document would. "
                "Competing answers would change what the model knows, which is a "
                "different experiment."
            ),
            "label_meaning": base.construction.get("label_meaning", ""),
            "llm_generated": False,
        },
    )
    _LOG.info(
        "built %s from %s: %d items, %s",
        name,
        base.eval_set_id,
        len(evalset),
        evalset.envelope_id,
    )
    return evalset
