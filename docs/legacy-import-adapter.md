# AMMS01 import staging and portable observation

## Two separate responsibilities

`src/ispec/legacy_export.py` is a standalone Python 3.8+ exporter. It needs only
the standard library, not MSPCRunner, pandas, iSPEC installation, or a database
driver. It implements the QUAL + QUANT merge / `e2g_` column-prefix conversion
previously embedded in MSPCRunner, with identity and merge validation. It also
converts an MS-file table with the `msf_` prefix. The old codebase is not invoked.

`packages/ispec-observer` is the portable read-only monitor. Its optional
`workflow` command watches exact file metadata and producer-written JSON job
receipts. AMMS01 needs no instrument configuration. Instrument export readers
remain separate adapters within that package, not requirements for every host.

Neither component runs grouping, concatenation, FileMaker imports, remote
commands, notifications, or scheduled tasks. No deployed service changes are
required for this first slice. The existing instrument-collector phone-home API
does not yet accept this generic workflow report; do not post it there or reuse
an instrument identity for AMMS01. Authenticated workflow ingestion and durable
outbound delivery are later work, independent of command authority.

## Historical compatibility candidate

Read-only inspection on 2026-10-07 found a small pair under
`D:/dataFusion_00_grouper` on AMMS01 (mounted locally at
`/home/alex/windows/amms01/d/dataFusion_00_grouper`):

| Artifact | Bytes | Rows | SHA-256 |
| --- | ---: | ---: | --- |
| `30896_2_1_none_e2g.tab` | 96332 | 93 | `a5cb3d5a7a0416591bee28f228e2568634c3657eb822a442b28325dba348b8cb` |
| `30896_2_1_none_msf.tab` | 550 | 1 | `76d1827c77b695e6015f4bff9f39316412991ddc2ba461b343398bf4321124e0` |

Both have source mtime `2019-10-09T14:15:55Z`. They are historical exports,
**not evidence of a current job or a successful FileMaker import**. Their contents
are ASCII, so this pair does not resolve non-ASCII encoding questions. A separate
`30896_2_1_none_0_msf.tab` also exists (540 bytes); the script's stored filename
field/mapping must determine which artifact a real import expects. The monitor
and test use the explicitly selected `none_msf` file, not filename discovery.

The local experiment-run import-status columns were unpopulated when inspected.
We have not verified this pair's import receipt directly in FileMaker. No source
data files are copied into Git and no PSM files are read for this trial.

## Validate or stage an export

Run the standalone file from this checkout, or copy that one file to a Python
3.8+ workstation. On Windows 7 select `py -3.8` explicitly rather than relying
on the default Python launcher selection. Explicit paths avoid the old working-directory /
`--source` problem. Example for modern, already-produced grouping outputs:

```powershell
py -3 legacy_export.py --record 30896 --run 2 --search 1 --label 0 --qual D:/grouped/30896_2_1_e2g_QUAL.tsv --quant D:/grouped/30896_2_1_none_e2g_QUANT.tsv --msf D:/grouped/30896_2_1_none_msf.tsv
```

Those modern input names are examples, not verified files on AMMS01. Default
operation validates and prints a manifest without writing anything. To produce
a new bundle, add `--write --output-parent D:/ispec-staging`, where that staging
directory already exists and is separate from the input/import directory.
Each call gets a new subdirectory and never overwrites an earlier bundle.
The manifest is written last; it records hashes, input paths, row counts, encoding,
and `staged_not_imported`. This is a staging receipt, not a FileMaker receipt.

For the historical pair, use `--combined` instead of QUAL/QUANT:

```powershell
py -3 legacy_export.py --record 30896 --run 2 --search 1 --label 0 --combined D:/dataFusion_00_grouper/30896_2_1_none_e2g.tab --msf D:/dataFusion_00_grouper/30896_2_1_none_msf.tab --e2g-filename 30896_2_1_none_e2g.tab --msf-filename 30896_2_1_none_msf.tab
```

Contract and deliberately conservative limits:

- Record/run/search/label are explicit. `none` and `0` normalize to numeric zero;
  the default output prefix uses all four components, with no fixed-width slicing.
  Explicit filename overrides accommodate FileMaker's existing filename fields.
- Each QUANT/combined input must contain one label. QUAL may describe a shared
  label pool, but it must contain the selected label. Mixed run identities fail.
- QUAL/QUANT join on record/run/search/GeneID/SRA. Duplicate or unmatched keys,
  overlapping columns with conflicting values, aliased duplicate headers, and
  malformed rows fail rather than silently multiply/drop rows. This strict merge
  may reject legitimate newer output shapes; review a real discrepancy before
  relaxing it, rather than silently filling missing values.
- `Symbol` maps to `GeneSymbol`; `LabelFLAG` maps to `EXPLabelFLAG`. Other
  columns are retained with the table prefix. Extra modern fields still require
  review against FileMaker's saved import mapping; staging is not compatibility
  certification. No quantification, SRA thresholds, or PSM acceptance rules change.
- Inputs are bounded to 64 MiB each and checked for size/mtime changes during read.
  Use stable completed exports. The adapter is not a transaction across concurrently
  changing source files; a producer completion receipt is the future handoff.
- Output defaults to strict Mac Roman because that is what the supplied FileMaker
  script requests. Unencodable text fails; explicit UTF-8 is available but requires
  a matching importer setting. Inputs default to UTF-8; historical Mac Roman needs
  `--input-encoding mac_roman`. LF line endings and column order are deterministic;
  byte hashes need not match an older CRLF export.
- The adapter never publishes into the legacy watched directory, writes FileMaker
  status flags, or imports e2g/PSM tables into local iSPEC. Supplying a staging path
  does not prove that path is unwatched: the operator must choose an isolated root.

## Windows monitoring trial

See the portable package's README and `examples/amms01-files.json`. Start with
the historical pair, one shot; then optionally foreground watch. The 0.1.2 AMMS01
configuration uses Alex's 60/120/300/600/1200/2400/3600/4800-second progression with
probabilistic early probes. Unchanged probes do not move the scheduled deadline;
detected changes reset to 60 seconds. Heartbeats are emitted on the first observation
after five minutes without an emission, without forcing extra reads. This is not
a fixed five-minute liveness guarantee. See the portable README for policy settings. It
only stats configured ordinary files. Optional
job receipts are bounded JSON; their contents are producer reports, never commands.
Stale files remain old evidence, not a success or health verdict. No folder-wide
scan of the very wide legacy directory is needed.

Python availability and execution on AMMS01 remain to be verified by the operator.
The local Linux test reads the mounted Windows files, not a remotely running agent.
The source package and portable ZIP support that manual trial. The 0.1.2 rsync
handoff copies code into D:/tools only; it does not install anything, register
startup tasks, or run programs on AMMS01.

## Next checkpoint before real importing

1. Verify the FileMaker restored find criteria, saved column mapping, and exact
   expected MS-file/e2g filenames, including label handling.
2. Stage one real modern gpGrouper output pair and compare field-level values with
   the historical contract. This slice tests format handling, not regrouping parity.
3. Run a separately approved FileMaker test import with per-import error checks,
   expected/actual counts, and a completion receipt. `AonSF` alone is not error
   checking. Do not mark `Import_EndFLAG` merely because files exist.
4. Later connect authenticated observation events to iSPEC, with event IDs,
   receipt provenance, offline retention, and restrained notifications. Keep
   retries of observation delivery separate from any retry of an import/job.

Tests: `python -m pytest tests/unit/test_legacy_export.py -q` and
`python -m unittest discover -s packages/ispec-observer/tests -v`.

Verification on 2026-10-07: 27 exporter tests and 41 observer tests passed together
under pytest (68 total). The rebuilt portable ZIP was extracted outside the repo;
its 41 tests passed there, and its isolated-mode CLI read both configured AMMS01
files through the existing mount. The real historical pair validated without any
writes (93 e2g rows, one MS-file row). Ruff passed for the new Python modules/tests.
This is Linux/source-format validation, not Windows deployment or FileMaker import
certification. The portable ZIP is generated under `packages/ispec-observer/dist/`;
build products remain Git-ignored, while code, examples, tests, and these notes are
available for a later scoped commit.

Python 3.8 / adaptive-polling follow-up (portable observer 0.1.1): 44 observer
tests and 28 standalone exporter tests passed under the installed Linux Python
3.8.20; the same 72 passed under Python 3.12. The extracted 0.1.1 ZIP passed its
44 tests under Python 3.8, and both the portable observer and standalone exporter
read/validated the same mounted historical pair without writing source files.
Exporter tests on 3.8 use `python3.8 -m pytest --noconftest
tests/unit/test_legacy_export.py -q`: this intentionally avoids loading unrelated
modern backend/DB fixtures. The full backend is not a Python 3.8 deployment target.
Windows 7 execution still needs the on-host check; no software was installed there.
