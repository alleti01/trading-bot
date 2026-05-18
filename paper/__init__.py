"""Paper-trading runtime: feeds → features → setups → risk → executor.

The orchestrator lives in :mod:`paper.loop`. This package intentionally
does not expose the loop class via ``__init__`` to keep imports explicit
at call sites.
"""
