"""The two-pane demo: state and decisions here, rendering in ``demo/``.

Grown every phase rather than built at the end (``TASKS.md`` Phase 2.5). The
centrepiece — Beat 4's matrix routing — needs Phases 4 and 5, and a demo
assembled the night before is how that beat ends up depending on work nobody
has run twice.
"""

from .session import DemoSession, PaneView, RequestOutcome
from .stream import Stream, StreamEvent, record_stream

__all__ = [
    "DemoSession",
    "PaneView",
    "RequestOutcome",
    "Stream",
    "StreamEvent",
    "record_stream",
]
