"""Bar-by-bar backtest engine. Day 4 deliverable.

Important fill convention (to be enforced):
- Setups fire on bar **close**.
- Fills happen on the **next bar's open** with slippage applied.
- Same-bar TP and SL ambiguity resolves SL-first (conservative).
"""
