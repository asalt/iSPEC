# Instrument monitoring roadmap

Alex's goals, captured 2026-11-20. There's no rush to release; the aim is that nothing gets forgotten. The current plan is in
[instrument-plugins-design-20261119.md](instrument-plugins-design-20261119.md),
and the Phase 1 evidence is in [instrument-pull-phase1-20261119.md](instrument-pull-phase1-20261119.md).

## In progress
- **Phase 2:** registry, read-only API and the real-data dev demo. Connection info (host/IP, share, mount path) can be edited in the UI and in a backend-local file, with no passwords. A dropped mount shows clearly as "source not mounted", never as idle.
- **Runs table on `/instrument/:id`:** the last 10 runs by default, a selector for how many to show, and a pager to move between pages.
- **Staff-only release on amms06:** same pattern as the OLMo cutover: staged release, nginx check, one-command rollback, and Alex gives the go.

## Next
- **Persistent mounts:** credential storage and auto-mount after a reboot or logout. Today the GVFS mounts vanish and need `mount_*` again. IPs can change after a PC restarts, which the existing `__mount_host_ip --save-only` store already handles.
- **Exploris (10.16.1.42) and Lumos (10.16.1.14):** mount their shares, then reuse the Thermo plugin.
- **Push adapters:** a small stdlib-first adapter on each instrument PC, with its SQLite as the outbox (design Phase 3).

## Scan-level QC views with mzio
- **Install mzio on amms06.** It isn't installed there yet; dev has `/home/alex/tools/mzio` at `85f048e`. Keep the two in sync.
- **Target HeLa QC runs.** The plugin already recognizes QC RAW files, for example 24 HeLa QCs on Eclipse D:/QC.
- **Thermo:** convert RAW to mzML with the ProteoWizard container, then run `mzio plot-survey` (TIC/BPC, MS1 map, DDA QC overview):
  `docker run -it --rm -e WINEDEBUG=-all -v $(pwd):/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest wine msconvert --filter 'peakPicking true 1-' file.raw`
- **Bruker:** no conversion is needed. `.d` holds `analysis.tdf` (SQLite), and mzio already reads native `.d`.
- **When to generate:** on request first, via a button in the instrument UI. Automatic generation for QC runs only, probably not for every RAW file.
- **Cache and storage:** store each output once by file fingerprint, and never write to the instrument share.

## A richer instrument page
- **Layout:** card or modal views (the "Pip-Boy" idea): status, the latest HeLa QC at a glance (TIC across time and other high-level plots), and recent runs.
- **Drill-down:** from a card into the full survey plots for one run.

## Later
- **Common-ion list:** curate the ions that show up again and again in QC runs.
- **Custom QC report:** build it on those ions, and track trends over time and across instruments.
