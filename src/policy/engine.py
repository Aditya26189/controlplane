"""The Rego adapter — the only thing in this repo that evaluates a policy rule.

``SPEC.md`` §7.1: *OPA/Rego or Cedar. Do not write a DSL.* We evaluate Rego,
through ``regopy`` (MIT, bindings to Microsoft's ``rego-cpp``), because it is a
pip-installable wheel rather than a 50 MB Go binary. Free-tier compute is a
hard constraint in ``CLAUDE.md`` and a policy engine that cannot be installed
there is a policy engine the demo does not have. ``DECISIONS.md`` 076 records
what that substitution does and does not buy, since ``rego-cpp`` is an
independent implementation of Rego and not OPA itself.

**An engine cannot be built from an unresolved bundle.** :class:`RegoEngine`
takes a :class:`~src.policy.resolution.ResolvedBundle`, not a
:class:`~src.policy.bundle.PolicyBundle`, so there is no ordering in which a
rule runs before its warrants were checked. That is ``SPEC.md`` §7.2 made
structural rather than remembered.

## Two traps in this binding, both silent

``set_input`` accepts anything. Handed a **JSON string** rather than a mapping
it does not raise — it sets the document to a string, every ``input.foo``
reference fails to resolve, every rule falls through to its default, and the
policy returns ``ALLOW`` for reasons no log will ever show. :meth:`decide`
refuses a non-mapping payload rather than passing it through.

Querying an entrypoint that does not exist raises a native exception through
the FFI boundary rather than a Python error. :meth:`__init__` therefore probes
the entrypoint at construction and converts the failure into a legible one,
because "the bundle names a rule the module does not define" must be a deploy
error and not a segfault under traffic.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Optional

from ..model.enums import Action
from .errors import BundleError, PolicyError
from .resolution import ResolvedBundle

__all__ = ["PolicyDecision", "RegoEngine"]


@dataclasses.dataclass(frozen=True)
class PolicyDecision:
    """What the rules decided, in the shape a certificate needs.

    Args:
        action: The decision.
        rule_id: The rule that fired. Never empty — a decision nobody can trace
            to a rule is a decision nobody can appeal, and the default rule
            counts as a rule.
        reason: Human-readable, shown to whoever is affected.
        policy_version: ``"<profile>/<version>"``, from the bundle.
        policy_hash: Content hash of manifest and rules together.
    """

    action: Action
    rule_id: str
    reason: str
    policy_version: str
    policy_hash: str

    def to_payload(self) -> dict:
        return {
            "action": self.action.value,
            "rule_id": self.rule_id,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
        }


#: Input used once at construction to prove the entrypoint exists and that the
#: module's default branch produces a well-formed decision. Deliberately empty
#: of every field the rules read, so it exercises the default and nothing else.
_PROBE: dict[str, Any] = {"probe": True}

#: Top-level keys the engine supplies from the resolved bundle. A payload
#: carrying one of these is refused rather than merged over: the first rule in
#: SPEC.md 7.2 keys on ``warrant.weakest_status``, and a request able to set
#: that field could assert its way past the rule the product rests on.
RESERVED_KEYS = ("profile", "policy_version", "warrant", "operating_point")


class RegoEngine:
    """Evaluates one resolved bundle's rules.

    Args:
        resolved: A bundle whose declared operating points are all backed by
            valid warrants. Taking the resolved type is the enforcement of
            ``SPEC.md`` §7.2 — an unresolved bundle has no path to a rule.

    Raises:
        BundleError: If the Rego module does not parse, the entrypoint is not
            defined, or the default decision is malformed.
    """

    def __init__(self, resolved: ResolvedBundle) -> None:
        from regopy import Interpreter  # imported here so the rest of the package loads without it

        self.resolved = resolved
        self.bundle = resolved.bundle
        self._interpreter = Interpreter()

        try:
            self._interpreter.add_module(self.bundle.name, self.bundle.rego_source)
        except Exception as exc:  # regopy raises RegoError, which is not an ImportError-safe name
            raise BundleError(
                f"bundle {self.bundle.name}/{self.bundle.version}: the Rego "
                f"module does not parse.\n{exc}"
            ) from exc

        # Prove the entrypoint resolves and the default branch is well formed.
        # A missing entrypoint surfaces through the FFI as a native exception,
        # so this is caught broadly and rewritten; a deploy-time error message
        # is the whole difference between a fixable typo and a crash in traffic.
        try:
            probe = self._evaluate(dict(_PROBE))
        except PolicyError:
            raise
        except Exception as exc:
            raise BundleError(
                f"bundle {self.bundle.name}/{self.bundle.version}: entrypoint "
                f"{self.bundle.entrypoint!r} could not be evaluated. The module "
                "must define it, and it must have a default branch that fires "
                f"when no rule matches.\n{type(exc).__name__}: {exc}"
            ) from exc

        self.default_decision = probe

    # -- evaluation --------------------------------------------------------- #

    def _evaluate(self, payload: Mapping[str, Any]) -> PolicyDecision:
        """Set input, query the entrypoint, and validate what comes back."""
        self._interpreter.set_input(dict(payload))

        output = self._interpreter.query(self.bundle.entrypoint)
        expressions = output.expressions()
        if not expressions:
            raise BundleError(
                f"entrypoint {self.bundle.entrypoint!r} produced no value. A "
                "policy that can decline to decide is a policy with an "
                "undefined action; give the rule a default branch."
            )
        raw = json.loads(expressions[0].json())
        return self._as_decision(raw)

    def _as_decision(self, raw: Any) -> PolicyDecision:
        """Validate the rules' output before anything acts on it."""
        if not isinstance(raw, Mapping):
            raise BundleError(
                f"the entrypoint must produce an object with 'action', "
                f"'rule_id' and 'reason'; got {type(raw).__name__}: {raw!r}"
            )
        missing = sorted({"action", "rule_id", "reason"} - set(raw))
        if missing:
            raise BundleError(
                f"the decision from {self.bundle.entrypoint!r} is missing "
                f"{missing}. rule_id and reason are not decoration: an action "
                "nobody can trace to a rule is an action nobody can appeal."
            )
        try:
            action = Action(str(raw["action"]))
        except ValueError as exc:
            raise BundleError(
                f"the rules produced action {raw['action']!r}, which is not one "
                f"of {[a.value for a in Action]}. A policy is refused rather "
                "than partially understood."
            ) from exc

        return PolicyDecision(
            action=action,
            rule_id=str(raw["rule_id"]),
            reason=str(raw["reason"]),
            policy_version=self.bundle.stamp["policy_version"],
            policy_hash=self.bundle.stamp["policy_hash"],
        )

    def decide(self, payload: Mapping[str, Any]) -> PolicyDecision:
        """Evaluate the rules against one request's facts.

        Args:
            payload: The request's own facts — the finding, the proposed action.
                The bundle's facts (``profile``, ``warrant``, ``operating_point``,
                ``policy_version``) are injected here and must not appear in it.
                **Must be a mapping.** A JSON
                string is refused rather than passed through: the binding
                accepts one, sets the input document to a string, and then every
                rule silently falls through to its default. That failure is
                invisible in the output and produces a permissive decision,
                which is the worst available combination.

        Returns:
            A :class:`PolicyDecision`.

        Raises:
            BundleError: If the payload is not a mapping, or the rules produce
                something that is not a well-formed decision.
        """
        if isinstance(payload, (str, bytes)):
            raise BundleError(
                "decide() takes a mapping, not a JSON string. The Rego binding "
                "accepts a string, sets the input document to it, and every "
                "rule then falls through to its default — a silent permissive "
                "decision with nothing in the output to show for it."
            )
        if not isinstance(payload, Mapping):
            raise BundleError(
                f"decide() takes a mapping, got {type(payload).__name__}"
            )
        supplied = sorted(set(payload) & set(RESERVED_KEYS))
        if supplied:
            raise BundleError(
                f"the request payload supplies {supplied}, which the engine "
                "injects from the resolved bundle. These are refused rather "
                "than overwritten: a request that can set "
                "warrant.weakest_status can assert its way past the one rule "
                "every other guarantee here rests on."
            )
        document = dict(payload)
        document.update(self.resolved.rego_facts())
        return self._evaluate(document)


def build_engine(resolved: ResolvedBundle) -> RegoEngine:
    """Construct an engine, with a legible error when the binding is absent.

    Args:
        resolved: The resolved bundle.

    Returns:
        A :class:`RegoEngine`.

    Raises:
        BundleError: If ``regopy`` is not installed. Never a fallback
            interpreter — a second evaluator with slightly different semantics
            is how "do not write a DSL" gets violated without anyone deciding to.
    """
    try:
        import regopy  # noqa: F401
    except ImportError as exc:
        raise BundleError(
            "the policy engine needs `regopy` (MIT, rego-cpp bindings): "
            "`pip install regopy`. There is deliberately no fallback "
            "interpreter — a second evaluator with slightly different "
            "semantics is how a hand-rolled DSL arrives without anyone "
            "choosing it (SPEC.md §7.1)."
        ) from exc
    return RegoEngine(resolved)
