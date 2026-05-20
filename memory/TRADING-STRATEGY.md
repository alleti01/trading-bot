# Trading Strategy

This document describes the bot's active playbook. Workflows read it at
pre-market; update it manually when you change rules or filters.

## Universe
- Trade only symbols listed in `ENABLED_SYMBOLS`.
- Respect per-symbol and portfolio trade caps from settings.

## Setup bias
- Prefer VWAP + EMA pullback continuation in the direction of the session trend.
- Skip counter-trend entries during the first 15 minutes unless ORB confirms.

## Risk
- One position at a time unless multi-symbol caps allow more.
- Default decision when uncertain: **hold**.
