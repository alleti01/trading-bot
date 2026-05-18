"""Order execution adapters.

The only adapter the Day-1 MVP can actually use is ``PaperExecutor`` (Day 5).
Live adapters are placeholders that raise on instantiation, so even with
``LIVE_ADAPTER_CONFIRMED=true`` the bot cannot route real orders until a
real adapter exists.
"""
