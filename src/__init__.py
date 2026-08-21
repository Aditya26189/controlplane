"""ControlPlane cascade-economics experiment.

Measures whether a linear probe on question-time activations can select which
LLM responses are worth sending to an expensive checker. The single output that
matters is ``lift = R / f`` — recall over measured flag rate.

Import order note: every module here is importable without a GPU. Anything that
needs CUDA (NF4 quantisation, activation extraction) fails at call time with a
clear message, not at import time, so the data/probe/economics/report stages
stay runnable on a laptop.
"""

__version__ = "0.1.0"
