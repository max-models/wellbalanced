"""The scope-profiler tag of wellbalanced's profiling regions.

The tag is registered as opt-in in :mod:`wellbalanced`, so the regions record nothing
unless an application enables it (``scope_profiler.setup(enable_tags=["wellbalanced"])``,
``SCOPE_PROFILER_ENABLE_TAGS=wellbalanced`` or ``scope-profiler run --enable-tag
wellbalanced``). Only coarse operations (solves, time steps, runs) have a region: a
disabled region still costs about 0.1 us per call.
"""

_PROFILE_TAGS = ("wellbalanced",)
