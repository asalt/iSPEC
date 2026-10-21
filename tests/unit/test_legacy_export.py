import csv
from dataclasses import FrozenInstanceError
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

# Exercise the standalone module without importing the modern iSPEC backend.
spec = importlib.util.spec_from_file_location(
    "ispec_legacy_export",
    Path(__file__).resolve().parents[2] / "src/ispec/legacy_export.py",
)
export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export
spec.loader.exec_module(export)


def test_identity_and_table_stay_frozen_and_keyword_only():
    with pytest.raises(TypeError):
        export.RunKey(1, 2, 3)
    with pytest.raises(TypeError):
        export.Table((), (), {})
    key = export.RunKey(record=1, run=2, search=3)
    with pytest.raises(FrozenInstanceError):
        key.record = 9
    with pytest.raises(FrozenInstanceError):
        export.Table(columns=(), rows=(), provenance={}).columns = ("other",)


def table(path, rows, *, prefix=""):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([prefix + name for name in rows[0]])
        writer.writerows(row.values() for row in rows)
    return path


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    identity = dict(EXPRecNo="58206", EXPRunNo="12", EXPSearchNo="34")
    gene = dict(**identity, GeneID="101", SRA="S")
    qual = dict(**gene, LabelFLAG="0", Symbol="GENE1", Description="Example protein")
    quant = dict(
        **gene, LabelFLAG="none", AreaSum_dstrAdj="125.25", iBAQ_dstrAdj="25.05"
    )
    msf = dict(RawFileName="sample.raw", **identity, PSMCount="25")
    return {
        "key": export.RunKey(record=58206, run=12, search=34),
        "qual": table(source / "qual.tsv", [qual]),
        "quant": table(source / "quant.tsv", [quant]),
        "msf": table(source / "msf.tsv", [msf]),
    }


def output_rows(raw, encoding="mac_roman"):
    return list(csv.DictReader(io.StringIO(raw.decode(encoding)), delimiter="\t"))


def change(path, **fields):
    rows = output_rows(path.read_bytes(), "utf-8")
    rows[0].update(fields)
    table(path, rows)


def test_split_export_preserves_identity_values_and_source_files(inputs):
    before = {name: inputs[name].read_bytes() for name in ("qual", "quant", "msf")}
    manifest, outputs = export.prepare(**inputs)
    assert set(outputs) == {"58206_12_34_0_e2g.tab", "58206_12_34_0_msf.tab"}
    row = output_rows(outputs["58206_12_34_0_e2g.tab"])[0]
    assert row["e2g_EXPLabelFLAG"] == "0"
    assert row["e2g_GeneSymbol"] == "GENE1"
    assert row["e2g_AreaSum_dstrAdj"] == "125.25"
    assert "e2g_LabelFLAG" not in row
    assert manifest["rows"] == {"e2g": 1, "msf": 1}
    assert manifest["filemaker_imported"] is None
    assert not manifest["filemaker_mapping_verified"]
    for name, original in before.items():
        assert inputs[name].read_bytes() == original
        assert len(manifest["sources"][name]["sha256"]) == 64


def test_qual_shared_label_pool_uses_selected_quant_label(inputs):
    change(inputs["qual"], LabelFLAG="1260;1270")
    change(inputs["quant"], LabelFLAG="1270")
    inputs["key"] = export.RunKey(record=58206, run=12, search=34, label=1270)
    _, outputs = export.prepare(**inputs)
    assert (
        output_rows(outputs["58206_12_34_1270_e2g.tab"])[0]["e2g_EXPLabelFLAG"]
        == "1270"
    )


@pytest.mark.parametrize(
    "side,fields",
    [
        ("qual", {"LabelFLAG": "1260;1270"}),
        ("quant", {"LabelFLAG": "1260;1270"}),
        ("quant", {"LabelFLAG": "1270"}),
        ("qual", {"GeneID": "other"}),
        ("quant", {"SRA": "A"}),
        ("qual", {"AreaSum_dstrAdj": "999"}),
        ("msf", {"EXPRecNo": "123"}),
        ("quant", {"EXPRunNo": "1.5"}),
        ("msf", {"RawFileName": ""}),
    ],
)
def test_bad_identity_join_or_overlap_fails_closed(inputs, side, fields):
    change(inputs[side], **fields)
    with pytest.raises(export.ExportError):
        export.prepare(**inputs)


@pytest.mark.parametrize("side", ["qual", "quant", "msf"])
def test_duplicate_rows_rejected(inputs, side):
    rows = output_rows(inputs[side].read_bytes(), "utf-8")
    table(inputs[side], rows + rows)
    with pytest.raises(export.ExportError):
        export.prepare(**inputs)


def test_extra_qual_row_is_not_silently_dropped(inputs):
    rows = output_rows(inputs["qual"].read_bytes(), "utf-8")
    table(inputs["qual"], rows + [{**rows[0], "GeneID": "102"}])
    with pytest.raises(export.ExportError, match="row loss"):
        export.prepare(**inputs)


def test_combined_prefixed_roundtrip_and_none_filename(inputs, tmp_path):
    _, outputs = export.prepare(**inputs)
    combined = tmp_path / "combined.tab"
    combined.write_bytes(outputs["58206_12_34_0_e2g.tab"])
    manifest, result = export.prepare(
        key=inputs["key"],
        combined=combined,
        msf=inputs["msf"],
        e2g_filename="58206_12_34_none_e2g.tab",
    )
    assert result["58206_12_34_none_e2g.tab"] == outputs["58206_12_34_0_e2g.tab"]
    assert "combined" in manifest["sources"]


@pytest.mark.parametrize(
    "name", ["../58206_12_34_0_e2g.tab", "58206_1_3_0_e2g.tab", "C:\\out.tab"]
)
def test_bad_output_names_rejected(inputs, name):
    with pytest.raises(export.ExportError):
        export.prepare(**inputs, e2g_filename=name)


def test_strict_encoding_never_silently_replaces_text(inputs):
    change(inputs["qual"], Description="\u57fa\u56e0")
    with pytest.raises(UnicodeEncodeError):
        export.prepare(**inputs)
    _, outputs = export.prepare(**inputs, output_encoding="utf-8")
    assert (
        output_rows(outputs["58206_12_34_0_e2g.tab"], "utf-8")[0]["e2g_Description"]
        == "\u57fa\u56e0"
    )


@pytest.mark.parametrize(
    "text",
    [
        "EXPRecNo\tEXPRunNo\tEXPSearchNo\n",
        "EXPRecNo\tEXPRunNo\tEXPSearchNo\n1\t2\n",
        "EXPRecNo\tEXPRunNo\tEXPSearchNo\tLabelFLAG\tEXPLabelFLAG\n1\t2\t3\t0\t0\n",
        'EXPRecNo\tEXPRunNo\tEXPSearchNo\tDescription\n1\t2\t3\t"line1\nline2"\n',
    ],
)
def test_malformed_empty_or_ambiguous_tables_rejected(tmp_path, text):
    path = tmp_path / "invalid.tsv"
    path.write_text(text)
    with pytest.raises(export.ExportError):
        export.read_table(path, prefix="e2g_")


def test_staging_separate_directory_and_repeated_runs_never_overwrite(inputs, tmp_path):
    manifest, outputs = export.prepare(**inputs)
    with pytest.raises(export.ExportError, match="separate staging"):
        export.stage(manifest, outputs, inputs["qual"].parent)
    parent = tmp_path / "staging"
    parent.mkdir()
    first = export.stage(manifest, outputs, parent)
    second = export.stage(manifest, outputs, parent)
    assert first != second
    assert {p.name for p in first.iterdir()} == {*outputs, "manifest.json"}
    receipt = json.loads((first / "manifest.json").read_text())
    assert receipt["status"] == "staged_not_imported"
    for name, raw in outputs.items():
        assert (first / name).read_bytes() == (second / name).read_bytes() == raw


def test_staging_failure_removes_only_its_own_incomplete_bundle(
    inputs, tmp_path, monkeypatch
):
    manifest, outputs = export.prepare(**inputs)
    parent = tmp_path / "staging"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("keep")

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(export.json, "dump", fail)
    with pytest.raises(OSError, match="full disk"):
        export.stage(manifest, outputs, parent)
    assert list(parent.iterdir()) == [sentinel]


def test_cli_is_standalone_and_default_does_not_write(inputs, tmp_path):
    script = Path(export.__file__).resolve()
    args = [
        sys.executable,
        "-I",
        str(script),
        "--record",
        "58206",
        "--run",
        "12",
        "--search",
        "34",
        "--label",
        "0",
    ]
    for key in ("qual", "quant", "msf"):
        args += ["--" + key, str(inputs[key])]
    before = set(tmp_path.rglob("*"))
    result = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "validated_not_written"
    assert set(tmp_path.rglob("*")) == before
    result = subprocess.run(
        args + ["--write"], cwd=tmp_path, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 1
    assert "output-parent" in json.loads(result.stdout)["error"]
