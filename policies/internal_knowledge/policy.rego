# internal_knowledge -- inline budget 500ms
#
# Balanced, retrieval verification on. The corpus exists and internal users
# tolerate friction.
#
# Rules are ordered by an explicit priority rather than by position in the file.
# Rego's complete rules conflict if two bodies hold at once, and resolving that
# by source order would make the policy depend on where somebody pasted a rule.
#
# The first rule is the entire product in four lines (SPEC.md 7.2): an
# irreversible action with no valid warrant behind it is blocked, regardless of
# what any detector said. It consults no score.
package controlplane

import rego.v1

default decision := {
	"action": "ALLOW",
	"rule_id": "R-default-allow",
	"reason": "no rule fired; nothing here licenses an action other than allow",
}

# The detector fired at THIS profile's operating point. The threshold is
# injected from the resolved warrant, so it is the measured one and not a
# number written into the policy.
fired if input.detector.score >= input.operating_point.threshold

candidates contains c if {
	input.warrant.weakest_status != "VALID"
	input.action.reversibility >= 2
	c := {
		"priority": 100,
		"action": "BLOCK",
		"rule_id": "R-no-warrant-irreversible",
		"reason": sprintf("no valid warrant backs an irreversible action: warrant status is %v", [input.warrant.weakest_status]),
	}
}

candidates contains c if {
	fired
	input.finding.category == "PII"
	input.finding.severity >= 3
	c := {
		"priority": 60,
		"action": "REDACT",
		"rule_id": "R-pii-redact",
		"reason": "high-severity PII finding at the validated operating point",
	}
}

# Balanced, with retrieval verification on. A flagged answer is held for
# verification against the corpus rather than escalated to a human: the corpus
# exists, and internal users tolerate the friction. REDACT is the action because
# what is withheld is the unverified claim, not the whole response.
candidates contains c if {
	fired
	input.finding.severity >= 3
	c := {
		"priority": 50,
		"action": "REDACT",
		"rule_id": "R-verify-before-answer",
		"reason": "detector fired at the validated operating point; unverified claim withheld pending retrieval check",
	}
}


priorities := [c.priority | some c in candidates]

decision := {"action": c.action, "rule_id": c.rule_id, "reason": c.reason} if {
	count(priorities) > 0
	some c in candidates
	c.priority == max(priorities)
}
