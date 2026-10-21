# Read-only instrument observations

This first slice is filesystem telemetry, not acquisition control or a vendor SDK.
It does not install services, restart iSPEC, modify credentials, send Slack, or
write anything to the observed directories. Ingestion is disabled until configured.
The separate `packages/ispec-observer` replay/measurement helper is unchanged.

## What is actually verified

On October 5, 2026 the operator's screenshot showed Bruker Compass HyStar 6.4,
nanoElute 2 and timsTOF Ultra. Those screen labels/measurements are a point-in-time
observation, not a discovered software interface. The existing Linux GIO mount
`~/windows/tims-ultra-0006/dshare` was readable. The `58799_58810` directory
contains project 1660 `.d` directories with `SampleInfo.xml`, `HyStarMetadata.xml`,
`analysis.tdf` and `analysis.tdf_bin`. No active vendor database was opened.
Windows Python, SDKs, Windows execution, and live pressure/flow interfaces have
not been verified. No package has been deployed to Windows by this change.

## Try a preview

From the backend checkout, this reads only metadata from the already-mounted,
explicitly selected historical 1660 directory and prints JSON. Nothing is sent:

```sh
python src/ispec/instruments/collector.py \
  --config configs/instrument-collector.bruker-mount.example.json
```

The collector is a single standard-library Python 3.10+ file, so it can be copied
to a Windows directory along with an operator-edited configuration. No pip
installation of the full backend is required. On that PC, after operator setup:

```powershell
py -3 collector.py --config instrument-collector.windows.local.json
```

Start from `configs/instrument-collector.windows.example.json`. Select the actual
acquisition folder; `D:\` is an example, not a verified Windows configuration.
A narrower folder is preferable to repeatedly surveying an entire disk.

The root must be explicit and absolute. The configured root may be a trusted
mount symlink, but discovered symlinks and Windows reparse points are skipped.
The collector never reads raw data, XML contents, vendor databases or logs.
It stats `.d` directories and five known metadata/data filenames only.
`observed_bytes` is the sum of those observed files, not the entire run size.
Directories may change during scanning; this is not an atomic acquisition snapshot.
The source root/configuration must be operator-controlled, not attacker-writable.

Scans run in a disposable child with an overall deadline (8 seconds by default),
entry and depth limits. Results contain the newest 20 runs **among those examined**;
truncation means we cannot promise the newest run on the whole disk was found.
`readable` means a directory was read, not that the intended share is mounted or
that an instrument is online. A stale mount exposing an empty local directory is
not evidence of idle acquisition. Mount verification remains an operator concern.

## Phone home, separately authorized

The new API provides:

- `POST /api/instruments/observations`: dedicated collector Bearer credential only.
- `GET /api/instruments/{instrument_id}/status`: existing staff/admin login only.

Configure `ISPEC_INSTRUMENT_COLLECTORS_FILE` to an **absolute** JSON path based on
`configs/instrument-collectors.example.json`. Each entry binds a collector ID,
allowed instrument IDs, and an environment-variable name for its dedicated secret.
Missing config/secrets disable ingestion; malformed config fails closed. Secrets
must be at least 32 characters and must not equal the general `ISPEC_API_KEY`.
Actual random secret provisioning and service reload remain explicit operator work.
Nothing in this change changes the optional authentication of older API routes.
Those older routes still need their normal API-key enforcement when exposed remotely.

On Windows, configure `ISPEC_INSTRUMENT_COLLECTOR_TOKEN` with that same dedicated
secret through the approved local environment. Do not put it in CLI arguments,
source control, screenshots, or the source JSON. To submit **one** observation:

```powershell
py -3 collector.py --config instrument-collector.windows.local.json --send --server https://YOUR-ISPEC-HOST
```

Use an existing approved HTTPS endpoint, or an explicitly approved tunnel to a
loopback HTTP endpoint. Plain HTTP to a LAN host is refused; no tunnel, TLS
termination, firewall changes, remote access or credential setup is performed by
the collector. Redirects and ambient HTTP proxies are disabled to keep the token
and metadata on the intended route. No command or Slack token is needed on Windows.

For an explicitly requested foreground loop, append `--watch --interval-seconds 60`.
Ctrl-C stops it; no daemon/task/service is installed. Default is one preview.
In watch/send mode an unconfirmed receipt retains the same event ID and payload
for the next interval. There is one in-memory pending observation, no offline
backlog; restarting loses that pending sample. Persistent scheduling/spooling and
retention are later decisions, not activated here. Do not run a permanent loop
until an observation retention policy has been agreed; there is no automatic prune.

## Evidence and storage

Each strict v1 JSON observation includes collector/instrument identity, event UUID,
UTC observation time, source kind/root/resolved root, source read status/errors,
bounded recent `.d` runs, last observed file names, sizes and mtimes, truncation
and collector version. State/progress/instrument error/heartbeat/pressure/flow
are explicitly unknown/null. Unsupported claims and command fields are rejected.
Source filenames can include research identifiers; staff access is intentional.

One additive `instrument_observation` table uses the existing **agent DB**. Its
unique event ID is reserved atomically with the payload; exact retries return the
original receipt and do not refresh heartbeat. Reusing an ID with different
evidence is a conflict (409). No generic agent command, relay or Slack job is
created. Existing DB initialization creates the new table when this version is
run; development tests use temporary databases, not the live database.

Latest status is selected by observation time, not HTTP arrival time. Collector
and source ages are separate; a new heartbeat with a days-old run stays days-old
source evidence. Missing source time remains unknown; future timestamps are
flagged and collector times over 60 seconds ahead are rejected. Buffered old
observations cannot displace newer status. A successful POST proves stored
telemetry only, not instrument health or acquisition completion.

Future verified log/SDK adapters can extend the versioned observation contract.
Do not infer completion from stable file size or percent from bytes. LC completion,
MS acquisition, DIA-NN, gpGrouper and site annotation are distinct stages. The
existing observer's historical pressure/flow replay must not become live readings.
Any later Slack output should use the existing allowlisted, receipt-backed relay.
Physical instrument commands remain out of scope and require separate operator
approval, authority, audit, allowlists and vendor safety interlocks.

## Separate Thermo observer pilot

The portable package now has a `thermo` command for bounded TNG log tails and
SHA-256 fingerprints of explicitly selected MSI tune/calibration XMB files.
See [the observer contract](../packages/ispec-observer/README.md#thermo-tng-logs-and-tunecalibration-fingerprints-source-checkout-pilot)
for configuration, parsing limits and the version-1 SQLite `file_revisions`
schema. The checkpoint holds recent log events; the dedicated local SQLite file
retains only changed XMB metadata/hashes, not binary contents or polling history.

This pilot is **not connected to the API contract above**, the agent DB, a
supervisor schedule, Slack, or instrument control. A later backend reader can
open its dedicated SQLite file read-only and check `observer_schema.version`.
Do not point the observer at an existing iSPEC database or assume that hashing
a calibration file proves calibration success or active instrument settings.

## Checks

```sh
python -m pytest -q --run-testclient \
  tests/unit/agent/test_instrument_observations.py tests/unit/api/test_instruments.py
```

These cover bounded metadata-only reads, portable isolated Python execution,
strict schemas, dedicated authentication/identity scope, staff-only queries,
concurrent deduplication, stale/out-of-order data and no command creation.
They do not certify Windows deployment or any vendor SDK.
