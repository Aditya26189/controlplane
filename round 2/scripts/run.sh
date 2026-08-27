#!/bin/sh
# Run a command and report its REAL exit status.
#
# `cmd | tail` reports *tail's* status, so a red suite reads as green. That
# happened twice in one session, the second time two messages after the trap was
# written into DECISIONS.md 050 -- because reading is not a control. This
# wrapper exists so the correct form is the default rather than the remembered
# one.
#
#   sh scripts/run.sh python -m pytest tests/ -q
#   RUN_TAIL=20 sh scripts/run.sh python scripts/03_matrix.py --config config.yaml
#
# Output goes to $RUN_LOG (default /tmp/run.log) so a long run can be tailed
# while it works, and the status is echoed before anything is piped.
set -u
log="${RUN_LOG:-/tmp/run.log}"
"$@" > "$log" 2>&1
status=$?
echo "EXIT=$status"
tail -n "${RUN_TAIL:-10}" "$log"
exit "$status"
