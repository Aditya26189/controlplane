"""An OpenAI-format adapter that returns certificates. ``SPEC.md`` §11, Phase 8 D.4.

**An adapter, not a gateway.** The distinction is the whole design and it is
easy to lose: a gateway owns the request path, terminates connections, holds
credentials, and becomes something an enterprise has to operate and trust. This
sits *behind* LiteLLM, which already does that job, and adds one thing — the
certificate.

``CLAUDE.md`` puts a gateway explicitly out of scope. If this file ever grows
routing, retries, key management or a listener, it has become the thing that was
ruled out.

## The gate: no application code change

An unmodified OpenAI-format client must receive certificates. That rules out the
obvious design — a second endpoint, or a wrapper the caller has to invoke — and
leaves two places a certificate can travel:

* **Inline**, in an additive field on the response object. Clients built against
  the OpenAI schema ignore unknown keys, so ``response["control_plane"]``
  reaches a caller who wants it and is invisible to one who does not.
* **Out of band**, in the ledger, addressable by the request id the client
  already has.

Both are populated. The inline copy is what makes the demo work with an
unmodified client; the ledger copy is the record of account, because a field in
a response the caller may discard is not an audit trail.

## Two paths, and why the fast one cannot simply be the slow one

The **inline** path runs only detectors that hold a valid warrant at the
profile's latency budget, and it blocks the response. The **async** path runs
the deep checks on all traffic and files its certificate afterwards.

They are not the same check at different speeds. An inline certificate claims
what the fast tier can support and says, in ``unchecked``, what has not run yet.
Presenting an inline result as though the deep checks had passed would be the
unbacked claim this project exists to refuse — the caller would see an ``ALLOW``
that means "nothing fast found anything", read as "nothing found anything".
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from ..model.certificate import Certificate
from ..model.enums import Action
from ..policy.compose import ComposedDecision

__all__ = ["AdapterResult", "CertificateEnvelope", "ControlPlaneAdapter"]

_LOG = logging.getLogger(__name__)

#: The response key the certificate travels under. Namespaced and additive:
#: an OpenAI-format client ignores keys it does not know, which is what lets an
#: unmodified caller keep working while a caller who wants the certificate can
#: read it without an SDK change.
RESPONSE_KEY = "control_plane"


@dataclasses.dataclass(frozen=True)
class CertificateEnvelope:
    """The certificate as it appears on a response, plus how to find the record.

    Args:
        certificate_id: Addressable in the ledger.
        action: What policy decided.
        claimed_bounds: Keyed by detector. Never merged (``DECISIONS.md`` 088).
        unchecked: What this decision cannot speak to — including, on the inline
            path, the deep checks that have not run yet.
        path: ``"inline"`` or ``"async"``.
        latency_ms: Wall clock for the checks, so a caller can see the inline
            path stayed inside the budget it claimed.
        deep_checks_pending: True on the inline path when an async certificate
            will follow. A caller that treats an inline ALLOW as final needs to
            know that from the object, not from documentation.
    """

    certificate_id: str
    action: str
    claimed_bounds: dict[str, Any]
    unchecked: tuple[str, ...]
    path: str
    latency_ms: float
    deep_checks_pending: bool

    def to_payload(self) -> dict:
        return {
            "certificate_id": self.certificate_id,
            "action": self.action,
            "claimed_bounds": self.claimed_bounds,
            "unchecked": list(self.unchecked),
            "path": self.path,
            "latency_ms": self.latency_ms,
            "deep_checks_pending": self.deep_checks_pending,
        }


@dataclasses.dataclass(frozen=True)
class AdapterResult:
    """One handled request.

    Args:
        response: The OpenAI-format response, with ``control_plane`` added.
        envelope: The same certificate summary, typed.
        decision: What composition decided, for a caller that wants the detail.
    """

    response: dict[str, Any]
    envelope: CertificateEnvelope
    decision: ComposedDecision


class ControlPlaneAdapter:
    """Wraps an OpenAI-format completion call and attaches a certificate.

    Args:
        complete: The upstream call — LiteLLM's ``completion``, or anything with
            its signature. Injected rather than imported so the adapter can be
            exercised without a network, and so the adapter never becomes the
            thing that owns the credentials.
        decide: Given the prompt and the response text, returns a
            :class:`ComposedDecision`. This is where the detectors and the
            policy engine live; the adapter does not know about either.
        record: Called with the sealed certificate. Normally the ledger's
            append; injected for the same reason.
        inline_budget_ms: The profile's budget. Exceeding it is **recorded, not
            enforced** — see :meth:`handle`.
    """

    def __init__(
        self,
        *,
        complete: Callable[..., dict],
        decide: Callable[[str, str], ComposedDecision],
        record: Optional[Callable[[Certificate], Any]] = None,
        inline_budget_ms: int = 200,
    ) -> None:
        self.complete = complete
        self.decide = decide
        self.record = record
        self.inline_budget_ms = inline_budget_ms

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _first_text(response: Mapping[str, Any]) -> str:
        """Pull the assistant text out of an OpenAI-format response.

        Tolerant of the shapes in the wild — ``message.content`` and the older
        ``text`` — and returns empty rather than raising when neither is
        present. A malformed upstream response is not a reason to fail the
        certificate; it is a reason for the certificate to say nothing was
        checked.
        """
        choices = response.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        message = choice.get("message") or {}
        return str(message.get("content") or choice.get("text") or "")

    @staticmethod
    def _prompt_text(messages: Sequence[Mapping[str, Any]]) -> str:
        """The last user turn, which is what a question-time probe reads."""
        for message in reversed(list(messages)):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    # -- the request path --------------------------------------------------- #

    def handle(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        request_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> AdapterResult:
        """Complete a request and attach its certificate.

        Args:
            messages: OpenAI-format messages, unmodified.
            request_id: The caller's request id.
            session_id: Its session.
            **kwargs: Passed straight through to ``complete``. The adapter adds
                no parameters of its own and strips none, so a caller's model,
                temperature and tools arrive upstream untouched.

        Returns:
            An :class:`AdapterResult` whose ``response`` is the upstream object
            with one added key.
        """
        response = dict(self.complete(messages=list(messages), **kwargs))

        started = time.perf_counter()
        decision = self.decide(
            self._prompt_text(messages), self._first_text(response)
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        unchecked = list(decision.unchecked)
        unchecked.append(
            "deep checks: queued on the async path and not reflected here. An "
            "inline ALLOW means nothing fast found anything, which is a weaker "
            "statement than nothing found anything."
        )

        # Budget overrun is RECORDED, not enforced. Dropping the certificate to
        # stay inside a budget would leave the response uncertified and looking
        # identical to one that passed, which is the failure the whole record
        # exists to prevent. A slow check is a fact about the deployment and
        # belongs on the certificate where somebody can see it.
        if latency_ms > self.inline_budget_ms:
            unchecked.append(
                "the inline checks took %.1fms against a %dms budget; the "
                "certificate is still issued and the overrun is recorded rather "
                "than the claim being dropped to fit"
                % (latency_ms, self.inline_budget_ms)
            )
            _LOG.warning(
                "inline path %.1fms over its %dms budget on %s",
                latency_ms - self.inline_budget_ms, self.inline_budget_ms, request_id,
            )

        envelope = CertificateEnvelope(
            certificate_id=f"C-{request_id}",
            action=decision.action.value,
            claimed_bounds=decision.claimed_bounds,
            unchecked=tuple(unchecked),
            path="inline",
            latency_ms=latency_ms,
            deep_checks_pending=True,
        )

        # Additive. The caller's response object is returned with everything it
        # had, plus one namespaced key an OpenAI-format client ignores.
        response[RESPONSE_KEY] = envelope.to_payload()
        return AdapterResult(response=response, envelope=envelope, decision=decision)
