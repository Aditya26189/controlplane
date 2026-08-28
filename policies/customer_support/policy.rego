# customer_support -- inline budget 200ms
#
# High precision, low flag rate. Over-blocking destroys the channel and the
# volume is high, so this profile acts on very little.
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

# No escalate-on-flag rule, deliberately. At a 10% flag rate on a high-volume
# channel, escalating every flagged item is a review queue nobody staffs, and
# over-blocking destroys the channel this profile exists to protect. The probe
# is a trigger; here it triggers redaction of PII and nothing else. Its
# correctness signal is carried on the certificate as a bound, not as an action.


priorities := [c.priority | some c in candidates]

decision := {"action": c.action, "rule_id": c.rule_id, "reason": c.reason} if {
	count(priorities) > 0
	some c in candidates
	c.priority == max(priorities)
}
