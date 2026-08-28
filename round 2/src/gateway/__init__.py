"""Sitting behind LiteLLM. ``SPEC.md`` §11.

An **adapter**, not a gateway. It does not own the request path, terminate
connections, hold credentials or route: LiteLLM already does that. It adds one
thing to a response that would otherwise be unchanged — the certificate.

``CLAUDE.md`` puts a gateway explicitly out of scope. If anything here grows a
listener, retries or key management, it has become the thing that was ruled out.
"""

from .adapter import RESPONSE_KEY, AdapterResult, CertificateEnvelope, ControlPlaneAdapter

__all__ = [
    "RESPONSE_KEY",
    "AdapterResult",
    "CertificateEnvelope",
    "ControlPlaneAdapter",
]
