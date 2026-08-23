"""A recorded request stream, so the demo does not depend on live traffic.

``DEMO.md`` requires the same stream to drive both panes. Recording it makes
that literal: one file, replayed identically into both, so a difference on
screen is a difference between the two *systems* and never between two samples.

It also makes the backup recording possible. ``DEMO.md``'s rule is that if the
live run fails you cut to a recording and say *"that's precisely the failure
mode we log"*; that only works if the recorded stream and the live stream are
the same artifact.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from ..model import content_hash
from ..validation.evalsets import EvalSet, ExtractionCache

__all__ = ["Stream", "StreamEvent", "record_stream"]


@dataclasses.dataclass(frozen=True)
class StreamEvent:
    """One request as it arrives at both panes.

    Args:
        index: Position in the stream.
        request_id: Stable id.
        session_id: Which session it belongs to. Sessions matter from Phase 9,
            where the Rule-of-Two flags are sticky per session.
        prompt: What was asked.
        response: What the model answered.
        item_id: The eval item this was drawn from, so a score can be looked up.
        row: Row index into the extraction cache.
        label: Ground truth, 1 meaning *incorrect*. Present because this is a
            recorded stream from a labelled set — it is **never shown to either
            pane**, only used to score the run afterwards.
        token_length: Prompt length, the envelope's highest-priority feature.
        segment: Which beat of the demo this event belongs to, e.g. ``"normal"``
            or ``"longctx"``.
    """

    index: int
    request_id: str
    session_id: str
    prompt: str
    response: str
    item_id: str
    row: int
    label: int
    token_length: int
    segment: str = "normal"


@dataclasses.dataclass(frozen=True)
class Stream:
    """A replayable sequence of requests.

    Args:
        stream_id: Content hash of the events, so two runs can be shown to have
            used the same stream.
        events: The requests, in order.
        source_eval_set: Which set they were drawn from.
        source_envelope_id: That set's envelope id.
    """

    stream_id: str
    events: tuple[StreamEvent, ...]
    source_eval_set: str
    source_envelope_id: str

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[StreamEvent]:
        return iter(self.events)

    def segment(self, name: str) -> tuple[StreamEvent, ...]:
        """Events belonging to one beat."""
        return tuple(e for e in self.events if e.segment == name)

    def save(self, path: str | Path) -> Path:
        """Write the stream as JSON, committed so a rehearsal is reproducible."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stream_id": self.stream_id,
            "source_eval_set": self.source_eval_set,
            "source_envelope_id": self.source_envelope_id,
            "events": [dataclasses.asdict(e) for e in self.events],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Stream":
        """Read a stream back."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            stream_id=data["stream_id"],
            events=tuple(StreamEvent(**e) for e in data["events"]),
            source_eval_set=data["source_eval_set"],
            source_envelope_id=data["source_envelope_id"],
        )


def record_stream(
    evalset: EvalSet,
    cache: ExtractionCache,
    *,
    n_events: int,
    seed: int,
    segment: str = "normal",
    session_size: int = 5,
    rows: Optional[np.ndarray] = None,
) -> Stream:
    """Draw a stream from an eval set's **test** rows.

    Test rows specifically: the demo shows the system operating on data the
    probe was not fitted on, which is the only setting in which the warrant's
    numbers describe what the audience is watching.

    Args:
        evalset: Set to draw from.
        cache: Its extraction, supplying token lengths.
        n_events: How many requests.
        seed: Sampling seed, so the stream is reproducible.
        segment: Beat label for these events.
        session_size: Requests per session id.
        rows: Explicit row indices to draw from. Defaults to the test split.

    Returns:
        A :class:`Stream`.
    """
    from ..validation.evalsets import TEST, split_by_question

    if rows is None:
        rows = split_by_question(evalset, seed=seed)[TEST]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(rows, size=min(n_events, rows.size), replace=False)
    chosen = np.sort(chosen)

    events = []
    for position, row in enumerate(chosen):
        item = evalset.items[int(row)]
        events.append(
            StreamEvent(
                index=position,
                request_id=f"R-{segment}-{position:04d}",
                session_id=f"S-{segment}-{position // max(1, session_size):03d}",
                prompt=item.prompt,
                response=item.response,
                item_id=item.item_id,
                row=int(row),
                label=int(item.label),
                token_length=int(cache.token_lengths[int(row)]),
                segment=segment,
            )
        )
    return Stream(
        stream_id=f"stream-{content_hash([dataclasses.asdict(e) for e in events])[:12]}",
        events=tuple(events),
        source_eval_set=evalset.eval_set_id,
        source_envelope_id=evalset.envelope_id,
    )
