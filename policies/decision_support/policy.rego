# decision_support -- inline budget 2s
#
# High recall, escalation-heavy. Low volume, high consequence, review
# affordable.
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

# Escalation-heavy. Low volume, high consequence, review affordable -- so the
# flag becomes a human's problem rather than a redaction. The uncertain band is
# separated from the rest because "the detector fired and is unsure" and "the
# detector fired confidently" are different queues, and merging them is how a
# review backlog stops being triaged.
candidates contains c if {
	fired
	input.finding.severity >= 3
	input.finding.confidence_band == "UNCERTAIN"
	c := {
		"priority": 55,
		"action": "ESCALATE",
		"rule_id": "R-escalate-uncertain",
		"reason": "high-severity finding in the uncertain band; routed to review",
	}
}

candidates contains c if {
	fired
	input.finding.severity >= 3
	c := {
		"priority": 50,
		"action": "ESCALATE",
		"rule_id": "R-escalate-on-flag",
		"reason": "detector fired at the validated operating point; routed to review",
	}
}


priorities := [c.priority | some c in candidates]

decision := {"action": c.action, "rule_id": c.rule_id, "reason": c.reason} if {
	count(priorities) > 0
	some c in candidates
	c.priority == max(priorities)
}
