---
name: ispec-dev-make
description: Control the iSPEC local dev stack via the top-level Makefile and tmux (start/stop/status/restart backend + supervisor + frontend + vLLM + Slack). Use when you need to manage running dev services, restart panes without reattaching, or inspect PID/state files for debugging.
---

# iSPEC Dev Make

## Quick Start

The top-level Makefile lives one directory above the backend repo (the `iSPEC/` folder). Run these from the repo root, or use `make -C .. ...` when you're inside `iSPEC/`.

Start the full dev tmux layout:

```bash
make dev-tmux
```

Show status (backend + supervisor):

```bash
make dev-status
```

Stop (backend + supervisor):

```bash
make dev-stop
```

Restart tmux panes (defaults to `backend supervisor` in session `ispecfull`):

```bash
make dev-restart
```

Restart a specific set of services in tmux:

```bash
make dev-restart DEV_RESTART_SERVICES="frontend vllm"
make dev-restart DEV_RESTART_SERVICES="slack"
```

Print PID files (one-line plaintext PIDs):

```bash
make dev-pids
```

## Common Tasks

Restart just one service in tmux:

```bash
make dev-restart DEV_RESTART_SERVICES=backend
make dev-restart DEV_RESTART_SERVICES=supervisor
```

Use a different tmux session:

```bash
make DEV_TMUX_SESSION=my-session dev-tmux
make DEV_TMUX_SESSION=my-session dev-restart
```

Run from inside `iSPEC/` without changing directories:

```bash
make -C .. dev-status
make -C .. dev-stop
make -C .. dev-restart
```

## Notes

- tmux window names are `backend`, `frontend`, `supervisor`, `vllm`, `slack` (see `scripts/dev-tmux.sh` at the repo root).
- State files live under `ISPEC_STATE_DIR` (dev default: repo-root `.pids/`):
  - `api_server.json`, `api_server.pid`
  - `supervisor.json`, `supervisor.pid`
