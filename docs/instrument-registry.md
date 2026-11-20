# Instrument registry and pull dashboard

The core DB owns registry settings. Staff sessions (editor/admin) can list
instruments, view status and paged runs, create an entry or patch it.
Collector tokens still authenticate only the v1 observation POST.
Registering a push instrument does not start a receiver for plugin batches.

For manual edits, set `ISPEC_DB_PATH` to the intended core DB. From the
backend checkout, using its Python environment:

```sh
umask 077
python scripts/instrument_registry.py fish-preview > instruments-initial.local.json
# Review the draft, then apply once to a NEW registry.
python scripts/instrument_registry.py apply instruments-initial.local.json
python scripts/instrument_registry.py export > instruments.local.json
# Edit in your editor, then apply. Export again after UI edits.
python scripts/instrument_registry.py apply instruments.local.json
```

Exports are editable snapshots, not a second live configuration. Revision
checks reject stale imports atomically. Fish preview asks the host's existing
profile/IP helpers for two known instruments without mounting, saving IPs,
or copying login fields. An edit changes monitoring, not fish configuration;
if a host moves, update the remount helper separately and remount as the reader
account. Both mount path and source root must agree. Pull sources are limited
to `~/windows` (or operator-set `ISPEC_INSTRUMENT_MOUNT_ROOT`). Host/share
identity is checked against the mounted filesystem. No passwords, commands,
automatic remounts, credential persistence or instrument control.

Set `ISPEC_INSTRUMENT_ARCHIVES` to a private local directory (default:
`instruments/` in the configured DB directory). Add the pinned observer
checkout's `src` to `PYTHONPATH` for the CLI only. Run once per instrument:

```sh
python scripts/pull_registered_instrument.py eclipse --thermo /path/to/thermo.json --qc /path/to/qc.json
python scripts/pull_registered_instrument.py tims-ultra-0006
```

The Thermo/QC files supply bounds, XMB names, telemetry fields and QC pattern;
registry roots select actual sources. The Bruker scan is depth-zero and limited
to the configured acquisition folder and LC tail. No timer is installed.
Changing a writer or source starts a new archive generation; previous files
remain intact. The dashboard reads only the current generation.

The API adapts the plugin header to the UI contract: scan time maps to
`last_collector_contact_at`, file mtime to `source_modified_at`, and
recorded bytes to `size_bytes`. First/last observation refer to stored
evidence, not acquisition times. The runs endpoint defaults to the most recent
10 distinct runs by source file modification time. Use `offset` and `limit`
(1-100), passing the first page's `snapshot` on later requests to keep the
order stable. `has_more` indicates another page; `truncated` indicates
incomplete scan coverage. Reload without a snapshot for newer observations. A missing
mount leaves last-seen runs visible; neither an old file nor a fresh heartbeat
means idle. Vendor-local times without an offset have null UTC. The list
isolates per-instrument archive errors. An unknown ID is 404; a registered
instrument with no contact has a null summary and an empty run page.
