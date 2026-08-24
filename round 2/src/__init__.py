"""ControlPlane — a warrant layer on top of AI detection.

A detector produces a score. A *warrant* is a separate, time-bounded,
evidence-backed statement about what that score is worth right now, on this
input distribution. The product is the three clauses in ``CLAUDE.md``:

    we tell you what your error rate is on your traffic,
    we tell you when that number stops being true,
    and we tell you what it costs to keep it true.

Import-order note: every module here must be importable without a GPU, without
network access, and without a model cache. Anything needing CUDA or a download
fails at call time with a clear message, never at import time, so the store,
validation, sampling, economics and policy layers stay runnable on a laptop and
in a test suite.
"""

__version__ = "0.1.0"
