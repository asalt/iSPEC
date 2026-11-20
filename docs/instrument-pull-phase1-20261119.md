# What the two mounts actually yielded

### Running the Bruker LC monitor on dev

From the wrapper checkout, this reads one bounded batch. Keep the same private
scratch directory across invocations so the plugin can resume from its offsets:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=iSPEC/src:iSPEC/packages/ispec-observer/src \
  python3 iSPEC/scripts/instrument_pull_pilot.py --bruker-only \
  --scratch /tmp/ispec-phase1-L8g12Sbw \
  --bruker iSPEC/configs/instrument-collector.bruker-mount.example.json \
  --bruker-lc-root "$HOME/windows/tims-ultra-0006/c/BDalSystemData/HyStar/LogFiles"
```

For a foreground check every minute on Linux, insert `watch -n 60 -x` before
`python3`; stop with Ctrl-C. For a new scratch location use
`mktemp -d /tmp/ispec-phase1-XXXXXXXX` once and substitute that path. The
first LC-capable reader upgrades an existing Bruker v1 scratch archive to v2
without resetting its run history or stream ID. Thermo archives stay v1.

The plugin selects the newest three dated **one- or two-column separation
execution logs** by filename time. Initial reads start at most 128 KiB from
EOF per file; repeat reads check the file identity and boundary fingerprints.
This is a rolling monitor: long gaps can miss older procedures, and starts
outside an initial tail are not reconstructed. It retains explicit
`STARTED`/`COMPLETED` transitions and the latest 24 measured
`act. pressure [bar]` samples. The label is matched exactly; setpoints,
gradient flow, and unlabelled final-channel pressures are not measurements.
Unsupported messages are ignored. No full log is stored.

**Verified Nov 19, 20:32 UTC:** three live files, six transitions and six
pressure samples, all matched independently at the archived source byte
offsets. Latest pressure: 800.013 bar at 11:06:22; latest LC completion:
11:39:41 (vendor local time). The next process read zero new payload bytes,
left the archive generation and row counts unchanged, and passed SQLite
`quick_check`. The earlier 12 Bruker MS run rows were untouched.

Read `coverage.lc` separately from `coverage.acquisition`. The plugin's
`lc.last_transition` and `lc.last_measured_pressure` include the source-local
timestamp, filename, byte offset and UTC recording time. They remain last-seen
observations during a share outage, not claims that the pressure is still
current, that the MS run succeeded, or that the instrument is idle.

### Earlier checks

First scan: Linux dev host (Alex's Pop!_OS session), 2026-11-19 06:21-06:25 UTC. The
documented source aliases were absent on amms06, and its GVFS directory had no
shares. Nothing was mounted or installed for this trial.

| Source | Verified evidence | Coverage limit |
|---|---|---|
| Eclipse D:/QC | 24 matching HeLa RAW files. Latest file mtime 2026-11-17 22:47:38Z; size 2,033,348,311 bytes. Size and nanosecond mtime matched a second stat on every file. | QC filename filter only; C: not mounted. No real TNG transitions, pressures or XMB fingerprints in this pilot. A file mtime is not proof of acquisition completion. |
| tims-ultra Dshare/58799_58810 | 12 run directories; latest observed file mtime 2026-10-02 00:54:12Z. Independently summed the known metadata/data file sizes and verified the recorded last-file path for all 12. | 86 entries examined, no read errors, truncated at depth zero. This is a historical acquisition folder, not a November-wide inventory. Bytes cover five named files if present, not every file in the run. |

Both instrument states are **unknown**, not idle. A healthy reader in November
can see an October run. Both first-scan summaries were partial; `last_successful_scan_at`
is null because it requires all configured sources to be readable. On final
repeat the persisted heartbeat times and fact counts did not move. `quick_check`
returned ok in both archives. Scratch: `/tmp/ispec-phase1-L8g12Sbw/{thermo,bruker}.sqlite`
on dev; temporary, not a backup. No legacy JSON outputs or production data writes.

**Repeat on the same archives, 06:59 UTC.** The Eclipse QC path now returned
`Invalid argument` from a host-side stat; the C: path was absent. The reason
for the D: failure is unknown. The plugin recorded one coverage change
(`partial` to `unavailable`, seq 28), retained all 24 prior RAW observations
and did not advance `latest_source_at`. Its zero files read on this check is
not an empty QC directory. Bruker still saw 12 runs in 86 entries, no read
errors, and depth-limited coverage. For all 12, a separate stat-only comparison
matched the stored bytes, last-file path and mtime (maximum of directory and
known files). No RAW contents were opened. A second unchanged invocation added
no rows to either DB. Final counts: Thermo 24 QC + 4 scan rows; Bruker 12 run +
2 scan rows. Both SQLite integrity quick checks passed; 175 focused tests
passed again. No claim of TNG rotation or live calibration verification.

**After the remount, 07:07 UTC.** On dev, `10.16.1.3/d` and
`10.16.1.63/dshare` were listable, neither instrument's C: SMB share was
present, and no new acquisition metadata appeared in either configured folder.
ECL QC recovered (24 matches; archive coverage back to partial). The two newest
matching RAWs, Nov 17 and Nov 2, both contained the UTF-16 string
`Orbitrap Eclipse` in the first 128 KiB; the Nov 17 header also had this value
beside `SiiXcalibur`. In the Bruker 58810 run, both XML sidecars and the
read-only `analysis.tdf` GlobalMetadata.InstrumentName said `timsTOF Ultra`.
These identify the recorded instruments in sampled files, not the current
physical attachment, PC login state, or latest run across the entire drive.

**C: shares restored, Nov 19 19:57 UTC.** Dev and the amms06 SSH account could
list both instrument C: shares and D:/Dshare. Direct file reads on amms06
confirmed the Eclipse DDI log and calibration XMB, and the Bruker HyStar
logbook. Only dev ran the scratch collector. These user-session mounts may
need an operator remount after reboot; a service account's access was not tested.

- **Eclipse:** all three configured roots readable. The first pull selected
  Nov 18/19 Firmware, DataSystem and DDI (six files; bounded tails, not a
  historical backfill). It recorded 370 non-telemetry log records, including
  53 acquisition-state records, 9 method starts and 8 finishes; these counts
  are not a completed-run tally. The Nov 19 DDI line at source-local
  13:10:23.3774 said `eVI_Running`, verified at its stored byte offset.
  DataSystem supplied source, collision, ion-gauge and UHV pressures; units
  and source timezone are unverified. The bounded sample window retained 120
  samples. Both selected XMBs matched independent SHA-256 and size checks
  (cal 874044 bytes, Nov 17 UTC mtime; tune 22028 bytes, Oct 13 UTC mtime).
  The 19:59 repeat read just 268 new Firmware bytes across the six files.
  `latest_source_at` can still show the Nov 17 QC mtime: unknown-zone TNG wall
  times are not inserted as UTC. Do not interpret that as a silent instrument.
- **Bruker:** the active sources are under `C:\BDalSystemData\HyStar\LogFiles`,
  including `CompassHyStarLogbook.log`, `Bruker OTOF MS.log`, and dated
  nanoElute separation execution logs. A Nov 19 one-column separation log
  explicitly reported `act. pressure [bar]` (800.013 at 11:06:22) and
  `COMPLETED` at 11:39:41, in source-local time. Gradient flow is programmed,
  not measured flow; pump procedure completion is not MS acquisition success.
  OTOF watchdog lines alone aren't run state either. These were bounded
  read-only source checks; the Bruker plugin still collects only the selected
  Dshare folder. Next candidate: tail the execution logs by offset for measured
  pressure and LC procedure transitions, separate from MS acquisition events.

**UI handoff (observed interface, not an API yet).** Status:
`instrument_id`, `plugin`, `stream_id`, `status` (coverage), `coverage` by
root alias, `scope` (bounds/counts/errors), `instrument_state`, `last_scan_at`
(last persisted check/heartbeat), `last_successful_scan_at`, `latest_source_at`.
Each `latest_runs` row has `root_alias`, `relative_path`, `source_kind`,
`observed_at` (filesystem mtime here), `recorded_at`, `observed_bytes`, and
nullable `run_state`. No run duration, inferred success or fabricated timestamps.
Times sort as fixed-width UTC; TNG source wall times with no verified timezone
stay local with null UTC. Mirror receipt and staleness policy belong to Phase 2/3.
On a status card show source coverage separately from instrument state; keep
last-seen runs visible on an unavailable source. Label `observed_at` here as
file modification time, not acquisition start/end. An immediate unchanged
poll need not move `last_scan_at`: it is the last **persisted** scan.

**Reproduce** from the wrapper root with locally available source configurations:

```bash
scratch=$(mktemp -d /tmp/ispec-phase1-XXXXXXXX)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=iSPEC/src:iSPEC/packages/ispec-observer/src \
  timeout 105 python3 iSPEC/scripts/instrument_pull_pilot.py --scratch "$scratch" \
  --thermo configs/eclipse-dev-stream.local.json --qc configs/eclipse-qc.local.json \
  --bruker iSPEC/configs/instrument-collector.bruker-mount.example.json
```

Repeat with the same scratch path for deduplication. Source reads use the existing
bounded workers before a SQLite write transaction; rows have one sequence per DB
and cursor advancement commits with evidence. An unchanged scan emits at most one
hourly heartbeat. The Bruker pilot calls the existing standalone scanner, not its
HTTP sender; Thermo uses its reader, not its JSON writer. 175 tests passed across
the observer suite, pilot, UTC, existing instrument API, agent store and collector
HTTP. Rotation, timeout and archive state tests are synthetic, distinct from the
36 real filesystem comparisons. Next useful evidence: re-run when Eclipse C: is
available and compare source lines and hashes directly. No UI branch or registry
changes; no instrument services.
