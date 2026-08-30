"""The LiteLLM adapter. ``SPEC.md`` §11, Phase 8 D.4.

The gate is that an unmodified OpenAI-format client receives certificates with
no application code change, so most of these tests are about what the adapter
does **not** do to a caller's request and response.
"""

from __future__ import annotations

import pytest

from controlplane.gateway import RESPONSE_KEY, ControlPlaneAdapter
from controlplane.model import Action, WarrantStatus
from controlplane.policy.compose import DetectorVerdict, compose

from .factories import make_warrant

MESSAGES = [
    {"role": "system", "content": "You are a banking assistant."},
    {"role": "user", "content": "Mera aadhaar 9999 0687 2026 hai, KYC update karo"},
]


def upstream(**kwargs) -> dict:
    """A minimal OpenAI-format response, with fields a caller would expect back."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": kwargs.get("model", "gpt-4o-mini"),
        "usage": {"prompt_tokens": 20, "completion_tokens": 7},
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Aapka KYC update ho gaya."},
            }
        ],
    }


def decide(prompt: str, response: str):
    return compose(
        [
            DetectorVerdict(
                detector_id="probe",
                status=WarrantStatus.VALID,
                fired=True,
                action=Action.ESCALATE,
                warrant=make_warrant(detector_id="probe", eval_set_id="envelope-a"),
            )
        ]
    )


def adapter(**kwargs) -> ControlPlaneAdapter:
    kwargs.setdefault("complete", upstream)
    kwargs.setdefault("decide", decide)
    return ControlPlaneAdapter(**kwargs)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_an_unmodified_client_gets_its_response_back_intact() -> None:
    """No application code change means nothing the caller relies on may move."""
    result = adapter().handle(messages=MESSAGES, request_id="R-1", session_id="S-1")
    for key, value in upstream().items():
        assert result.response[key] == value, f"{key} was altered"


def test_the_certificate_arrives_on_an_additive_namespaced_key() -> None:
    """An OpenAI-format client ignores keys it does not know, which is what lets
    the certificate reach a caller who wants it and no one else."""
    result = adapter().handle(messages=MESSAGES, request_id="R-1", session_id="S-1")
    assert RESPONSE_KEY in result.response
    assert set(result.response) - set(upstream()) == {RESPONSE_KEY}
    payload = result.response[RESPONSE_KEY]
    assert payload["certificate_id"] == "C-R-1"
    assert payload["action"] == Action.ESCALATE.value
    assert payload["claimed_bounds"]


def test_caller_parameters_pass_through_untouched() -> None:
    """The adapter adds no parameters and strips none."""
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return upstream(**kwargs)

    adapter(complete=capture).handle(
        messages=MESSAGES,
        request_id="R-1",
        session_id="S-1",
        model="claude-3",
        temperature=0.2,
        tools=[{"type": "function"}],
    )
    assert seen["model"] == "claude-3"
    assert seen["temperature"] == 0.2
    assert seen["tools"] == [{"type": "function"}]
    assert seen["messages"] == MESSAGES


# --------------------------------------------------------------------------- #
# What the inline certificate may and may not claim
# --------------------------------------------------------------------------- #


def test_an_inline_certificate_says_the_deep_checks_have_not_run() -> None:
    """An inline ALLOW means "nothing fast found anything", which is weaker than
    "nothing found anything". The difference has to be on the object."""
    result = adapter().handle(messages=MESSAGES, request_id="R-1", session_id="S-1")
    assert result.envelope.path == "inline"
    assert result.envelope.deep_checks_pending is True
    assert any("deep checks" in note for note in result.envelope.unchecked)


def test_a_budget_overrun_is_recorded_rather_than_dropping_the_claim() -> None:
    """Dropping the certificate to stay inside a budget leaves the response
    uncertified and looking identical to one that passed."""
    import time

    def slow(prompt, response):
        time.sleep(0.02)
        return decide(prompt, response)

    result = adapter(decide=slow, inline_budget_ms=1).handle(
        messages=MESSAGES, request_id="R-1", session_id="S-1"
    )
    assert RESPONSE_KEY in result.response
    assert result.envelope.latency_ms > 1
    assert any("budget" in note for note in result.envelope.unchecked)


def test_bounds_reach_the_response_keyed_by_detector() -> None:
    """``DECISIONS.md`` 088: bounds are never merged, including on the way out."""
    result = adapter().handle(messages=MESSAGES, request_id="R-1", session_id="S-1")
    bounds = result.response[RESPONSE_KEY]["claimed_bounds"]
    assert set(bounds) == {"probe"}
    assert bounds["probe"]["detector_id"] == "probe"


# --------------------------------------------------------------------------- #
# Robustness of the read path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{"index": 0, "message": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "text": "older completion shape"}]},
        {},
    ],
)
def test_an_unusual_upstream_shape_still_gets_a_certificate(response) -> None:
    """A malformed upstream response is not a reason to fail the certificate.

    It is a reason for the certificate to say nothing was checked — which is a
    statement, where a missing certificate is an absence nobody notices.
    """
    result = adapter(complete=lambda **k: dict(response)).handle(
        messages=MESSAGES, request_id="R-1", session_id="S-1"
    )
    assert RESPONSE_KEY in result.response


def test_the_prompt_read_is_the_last_user_turn() -> None:
    """A question-time probe reads the question, not the system prompt."""
    seen = {}

    def capture(prompt, response):
        seen["prompt"] = prompt
        return decide(prompt, response)

    adapter(decide=capture).handle(messages=MESSAGES, request_id="R-1", session_id="S-1")
    assert seen["prompt"] == MESSAGES[-1]["content"]


def test_the_adapter_owns_no_credentials_and_no_routing() -> None:
    """``CLAUDE.md`` rules a gateway out of scope. This asserts the shape stays
    an adapter: the upstream call is injected, so it never holds a key."""
    import inspect

    source = inspect.getsource(ControlPlaneAdapter)
    for forbidden in ("api_key", "base_url", "requests.", "httpx.", "retry"):
        assert forbidden not in source, (
            f"{forbidden!r} appears in the adapter; it has started becoming the "
            "gateway CLAUDE.md rules out"
        )
