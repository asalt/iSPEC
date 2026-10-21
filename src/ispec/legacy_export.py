"""Standalone, standard-library gpGrouper -> FileMaker staging adapter.

This module never connects to FileMaker or changes import/grouping flags.
Run it directly with Python; importing iSPEC or MSPCRunner is unnecessary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile


VERSION = "0.1"
RUN_FIELDS = ("EXPRecNo", "EXPRunNo", "EXPSearchNo")
JOIN_FIELDS = (*RUN_FIELDS, "GeneID", "SRA")


class ExportError(ValueError):
    pass


def integer(value: str, *, label: bool = False) -> str:
    if label and value.lower() == "none":
        return "0"
    try:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value():
            raise ExportError("Identity fields must be integers.")
        if number < (0 if label else 1) or number > 2**63 - 1:
            raise ExportError("Identity field is out of range.")
        return str(int(number))
    except InvalidOperation as exc:
        raise ExportError("Identity fields must be integers (or label=none).") from exc


@dataclass(frozen=True, init=False)
class RunKey:
    record: int
    run: int
    search: int
    label: int = 0

    # Explicit keyword-only construction preserves the contract on Python 3.8.
    def __init__(self, *, record: int, run: int, search: int, label: int = 0):
        for name, value in (
            ("record", record),
            ("run", run),
            ("search", search),
            ("label", label),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExportError("RunKey requires integer identifiers.")
            integer(str(value), label=name == "label")
            object.__setattr__(self, name, value)

    @property
    def prefix(self) -> str:
        return f"{self.record}_{self.run}_{self.search}"

    @property
    def sample(self) -> str:
        return f"{self.prefix}_{self.label}"


@dataclass(frozen=True, init=False)
class Table:
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    provenance: dict

    def __init__(
        self,
        *,
        columns: tuple[str, ...],
        rows: tuple[dict[str, str], ...],
        provenance: dict,
    ):
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "provenance", provenance)


def read_table(path: str | Path, *, prefix: str, encoding: str = "utf-8-sig") -> Table:
    path = Path(path).resolve(strict=True)
    before = path.stat()
    limit = 64 * 1024 * 1024
    if not path.is_file() or before.st_size > limit:
        raise ExportError("Input must be a regular table no larger than 64 MiB.")
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    after = path.stat()
    if len(raw) > limit or (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExportError("Input changed during reading or exceeded the size limit.")
    reader = csv.reader(
        io.StringIO(raw.decode(encoding), newline=""), delimiter="\t", strict=True
    )
    header = next(reader, [])
    columns = []
    for column in header:
        name = column[len(prefix) :] if column.startswith(prefix) else column
        name = {"LabelFLAG": "EXPLabelFLAG", "Symbol": "GeneSymbol"}.get(name, name)
        if (
            not name
            or not name.replace("_", "").isascii()
            or not name.replace("_", "").isalnum()
        ):
            raise ExportError("Invalid column name.")
        if name in columns:
            raise ExportError(f"Duplicate/aliased column: {name}")
        columns.append(name)
    if not set(RUN_FIELDS) <= set(columns):
        raise ExportError("Table is missing record/run/search columns.")
    rows = []
    for values in reader:
        if not values:
            continue
        if len(values) != len(columns):
            raise ExportError("Ragged table row.")
        if any(any(c in value for c in ("\t", "\r", "\n", "\0")) for value in values):
            raise ExportError(
                "Embedded tabs/newlines/NULs are not supported by this import contract."
            )
        row = dict(zip(columns, values))
        for name in RUN_FIELDS:
            row[name] = integer(row[name])
        rows.append(row)
    if not rows:
        raise ExportError("Empty input table.")
    return Table(
        columns=tuple(columns),
        rows=tuple(rows),
        provenance={
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "rows": len(rows),
            "encoding": encoding,
        },
    )


def validate_run(table: Table, key: RunKey) -> None:
    expected = tuple(map(str, (key.record, key.run, key.search)))
    if any(tuple(row[name] for name in RUN_FIELDS) != expected for row in table.rows):
        raise ExportError("Input contains a different record/run/search identity.")


def gene_key(row: dict[str, str]) -> tuple[str, ...]:
    if any(not row.get(name) for name in JOIN_FIELDS):
        raise ExportError("Missing gene join identity.")
    return tuple(row[name] for name in JOIN_FIELDS)


def join_e2g(qual: Table, quant: Table, key: RunKey) -> Table:
    """QUAL may advertise a semicolon-separated label pool; QUANT selects one label."""
    validate_run(qual, key)
    validate_run(quant, key)
    if "EXPLabelFLAG" not in quant.columns:
        raise ExportError("QUANT must explicitly identify its label.")
    index = {}
    for row in qual.rows:
        identity = gene_key(row)
        if identity in index:
            raise ExportError("Duplicate QUAL join key; refusing a many-to-many merge.")
        if "EXPLabelFLAG" in row:
            labels = {
                integer(value, label=True) for value in row["EXPLabelFLAG"].split(";")
            }
            if str(key.label) not in labels:
                raise ExportError("Requested label is absent from QUAL label pool.")
        index[identity] = row
    rows, used = [], set()
    for source in quant.rows:
        identity = gene_key(source)
        if identity in used or identity not in index:
            raise ExportError("Duplicate or unmatched QUANT join key.")
        if integer(source["EXPLabelFLAG"], label=True) != str(key.label):
            raise ExportError("QUANT must contain only the explicitly selected label.")
        used.add(identity)
        row = {**source, "EXPLabelFLAG": str(key.label)}
        for column, value in index[identity].items():
            if column == "EXPLabelFLAG":
                continue
            if column in row and row[column] != value:
                raise ExportError(f"Conflicting QUAL/QUANT values for {column}.")
            row[column] = value
        rows.append(row)
    if used != set(index):
        raise ExportError("Unmatched QUAL rows; refusing silent inner-join row loss.")
    columns = tuple(dict.fromkeys((*quant.columns, *qual.columns)))
    return Table(columns=columns, rows=tuple(rows), provenance={})


def validate_e2g(table: Table, key: RunKey) -> Table:
    validate_run(table, key)
    required = {*JOIN_FIELDS, "EXPLabelFLAG", "AreaSum_dstrAdj", "iBAQ_dstrAdj"}
    if not required <= set(table.columns):
        raise ExportError(
            f"Missing e2g fields: {sorted(required - set(table.columns))}"
        )
    seen, rows = set(), []
    for source in table.rows:
        identity = gene_key(source)
        if identity in seen:
            raise ExportError("Duplicate e2g join key.")
        seen.add(identity)
        label = integer(source["EXPLabelFLAG"], label=True)
        if label != str(key.label):
            raise ExportError("Combined e2g contains a different label.")
        rows.append({**source, "EXPLabelFLAG": label})
    return Table(columns=table.columns, rows=tuple(rows), provenance=table.provenance)


def encode_table(table: Table, prefix: str, encoding: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerow([prefix + name for name in table.columns])
    writer.writerows([[row[name] for name in table.columns] for row in table.rows])
    return output.getvalue().encode(encoding, errors="strict")


def prepare(
    *,
    key: RunKey,
    msf: Path,
    qual: Path | None = None,
    quant: Path | None = None,
    combined: Path | None = None,
    input_encoding: str = "utf-8-sig",
    output_encoding: str = "mac_roman",
    e2g_filename: str | None = None,
    msf_filename: str | None = None,
) -> tuple[dict, dict[str, bytes]]:
    if output_encoding not in {"mac_roman", "utf-8"}:
        raise ExportError("Choose an explicit supported output encoding.")
    if (combined is not None) == (qual is not None or quant is not None):
        raise ExportError("Choose combined e2g OR both QUAL and QUANT.")
    sources = {}
    if combined is not None:
        e2g = read_table(combined, prefix="e2g_", encoding=input_encoding)
        sources["combined"] = e2g.provenance
    else:
        if qual is None or quant is None:
            raise ExportError("Both QUAL and QUANT are required.")
        quality = read_table(qual, prefix="e2g_", encoding=input_encoding)
        quantity = read_table(quant, prefix="e2g_", encoding=input_encoding)
        sources.update(qual=quality.provenance, quant=quantity.provenance)
        e2g = join_e2g(quality, quantity, key)
    e2g = validate_e2g(e2g, key)
    files = read_table(msf, prefix="msf_", encoding=input_encoding)
    validate_run(files, key)
    names = [row.get("RawFileName", "") for row in files.rows]
    if not all(names) or len(names) != len(set(names)):
        raise ExportError("MS-file rows need unique, nonempty RawFileName values.")
    sources["msf"] = files.provenance
    e2g_filename = e2g_filename or f"{key.sample}_e2g.tab"
    msf_filename = msf_filename or f"{key.sample}_msf.tab"
    outputs = {}
    for name, table, prefix in (
        (e2g_filename, e2g, "e2g_"),
        (msf_filename, files, "msf_"),
    ):
        if (
            not name.startswith(key.prefix + "_")
            or not name.endswith(".tab")
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for c in name
            )
            or name in outputs
        ):
            raise ExportError(
                "Output names must be distinct .tab basenames with the exact run prefix."
            )
        outputs[name] = encode_table(table, prefix, output_encoding)
    manifest = {
        "schema_version": 1,
        "adapter_version": VERSION,
        "sample": key.sample,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "filemaker_imported": None,
        "filemaker_mapping_verified": False,
        "encoding": output_encoding,
        "sources": sources,
        "rows": {"e2g": len(e2g.rows), "msf": len(files.rows)},
        "outputs": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in outputs.items()
        },
    }
    return manifest, outputs


def stage(manifest: dict, outputs: dict[str, bytes], parent: Path) -> Path:
    parent = parent.resolve(strict=True)
    for source in manifest["sources"].values():
        source_root = Path(source["path"]).parent
        if parent == source_root or source_root in parent.parents:
            raise ExportError(
                "Use a separate staging root, not an input/import directory."
            )
    folder = Path(tempfile.mkdtemp(prefix=manifest["sample"] + "-", dir=parent))
    try:
        for name, raw in outputs.items():
            with (folder / name).open("xb") as handle:
                handle.write(raw)
        # Written last: absence of this receipt means the bundle is incomplete.
        with (folder / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump({**manifest, "status": "staged_not_imported"}, handle, indent=2)
    except BaseException:
        shutil.rmtree(folder)
        raise
    return folder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("record", "run", "search", "label"):
        parser.add_argument("--" + name, type=int, required=True)
    for name in ("qual", "quant", "combined", "msf", "output-parent"):
        parser.add_argument("--" + name, type=Path, required=name == "msf")
    parser.add_argument(
        "--input-encoding", choices=("utf-8-sig", "mac_roman"), default="utf-8-sig"
    )
    parser.add_argument(
        "--output-encoding", choices=("mac_roman", "utf-8"), default="mac_roman"
    )
    parser.add_argument("--e2g-filename")
    parser.add_argument("--msf-filename")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write a new staging bundle; default validates only.",
    )
    args = parser.parse_args(argv)
    try:
        if args.write and args.output_parent is None:
            raise ExportError("--write requires an existing --output-parent.")
        key = RunKey(
            record=args.record, run=args.run, search=args.search, label=args.label
        )
        manifest, outputs = prepare(
            key=key,
            msf=args.msf,
            qual=args.qual,
            quant=args.quant,
            combined=args.combined,
            input_encoding=args.input_encoding,
            output_encoding=args.output_encoding,
            e2g_filename=args.e2g_filename,
            msf_filename=args.msf_filename,
        )
        result = {**manifest, "status": "validated_not_written"}
        if args.write:
            result.update(
                status="staged_not_imported",
                directory=str(stage(manifest, outputs, args.output_parent)),
            )
        print(json.dumps(result, indent=2))
        return 0
    except (ExportError, OSError, UnicodeError, csv.Error) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
