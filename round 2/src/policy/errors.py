"""Failures a policy bundle can have. ``SPEC.md`` §7.2.

Separated from the modules that raise them so that a caller can catch
"this bundle is malformed" apart from "this bundle asks for a claim nobody can
back", which are different operational problems with different owners: the
first is a policy author's typo, the second is a statement that the detector
fleet cannot currently support a deployment somebody wants.
"""

from __future__ import annotations

__all__ = ["BundleError", "PolicyError", "WarrantResolutionError"]


class PolicyError(Exception):
    """Base for everything this package raises."""


class BundleError(PolicyError):
    """The bundle is malformed, unparseable, or internally inconsistent."""


class WarrantResolutionError(PolicyError):
    """The bundle is well-formed and asks for a claim the matrix cannot back.

    **Never a warning** (``DECISIONS.md`` 012, ``config.policy
    .fail_closed_on_missing_warrant``). Warning and continuing is the ordinary
    engineering choice and it silently reintroduces exactly the unbacked claim
    the product exists to refuse: a profile would go live quoting bounds that
    no valid warrant supplies, and nothing downstream would ever know.

    Args:
        message: The full explanation, which must name the warrant key.
        failures: One line per unmet requirement, in declaration order.
    """

    def __init__(self, message: str, failures: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.failures = failures
