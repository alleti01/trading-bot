# Deploying the autonomous intraday loop

The loop is a long-running process. It scans every
`WORKFLOW_SCAN_INTERVAL_MINUTES` during market hours and idles otherwise,
so you start it **once** and it handles every session — as long as the
host stays on and awake during 9:30–16:00 ET.

## Option 1 — macOS (your Mac), manual

```bash
deploy/run_intraday.sh                 # live paper (places orders)
deploy/run_intraday.sh --workflow-dry-run   # dry run (no orders)
```

Keep the terminal open and the lid up (closing the lid sleeps the Mac
despite `caffeinate`).

## Option 2 — macOS, auto-start with launchd

```bash
cp deploy/com.tradeify.intraday.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.tradeify.intraday.plist   # starts now + at login
launchctl unload ~/Library/LaunchAgents/com.tradeify.intraday.plist # stop
```

Auto-restarts on crash/login. The Mac still must be awake during the
session.

## Option 3 — always-on Linux PC, systemd (recommended for hands-off)

```bash
sudo cp deploy/tradeify-intraday.service /etc/systemd/system/
# edit User / paths in the file first
sudo systemctl daemon-reload
sudo systemctl enable --now tradeify-intraday
journalctl -u tradeify-intraday -f
```

Survives reboots and crashes; the Mac becomes irrelevant.

## Notes

- The `.env` (with keys) lives in the working directory and is **not**
  committed. Copy it to the host securely.
- Check status anytime with `python -m app.main --paper-report`.
- LIVE remains locked everywhere; this only ever runs paper.
