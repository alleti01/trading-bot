"""LLM-powered advisory agents.

INVARIANTS enforced elsewhere by tests:
- Agents MUST NOT import ``execution`` or ``risk``.
- Agents MUST NOT place trades, change risk limits, or override decisions.
- Agent output MUST validate against a Pydantic schema; failed parse is
  logged and discarded.
"""
