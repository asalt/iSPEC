from __future__ import annotations

import json
from pathlib import Path

from ispec.db.audit import build_legacy_gap_rows, write_import_audit_artifacts
from ispec.db.connect import get_session
from ispec.db.models import Project
from ispec.omics.connect import get_omics_session


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_legacy_gap_rows_classifies_mapped_and_file_imported(tmp_path):
    schema_path = tmp_path / "schema.json"
    mapping_path = tmp_path / "mapping.json"
    tables_path = tmp_path / "tables.tsv"

    _write_text(
        schema_path,
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "fields": ["prj_PRJRecNo", "prj_ModificationTS"],
                        "pk_guess": "prj_PRJRecNo",
                        "modification_ts_guess": "prj_ModificationTS",
                    }
                }
            }
        ),
    )
    _write_text(
        mapping_path,
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "local_table": "project",
                        "pk": {"legacy": "prj_PRJRecNo"},
                        "modified_ts": "prj_ModificationTS",
                    }
                }
            }
        ),
    )
    _write_text(
        tables_path,
        "tables\n"
        "iSPEC_Projects\n"
        "BCM_iSPEC_psms_expruns\n",
    )

    rows = build_legacy_gap_rows(
        legacy_schema_path=str(schema_path),
        legacy_mapping_path=str(mapping_path),
        legacy_tables_file_path=str(tables_path),
    )
    by_name = {row["legacy_table"]: row for row in rows}

    assert by_name["iSPEC_Projects"]["classification"] == "mapped"
    assert by_name["iSPEC_Projects"]["current_ingest_path"] == "sync-legacy-projects"
    assert by_name["BCM_iSPEC_psms_expruns"]["classification"] == "file_imported"
    assert by_name["BCM_iSPEC_psms_expruns"]["current_ingest_path"] == "import-psm"
    assert "Field metadata missing" in by_name["BCM_iSPEC_psms_expruns"]["blocker"]


def test_write_import_audit_artifacts_writes_json_and_tsv(tmp_path):
    core_db = tmp_path / "core.db"
    out_dir = tmp_path / "out"
    schema_path = tmp_path / "schema.json"
    mapping_path = tmp_path / "mapping.json"
    tables_path = tmp_path / "tables.tsv"

    _write_text(
        schema_path,
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "fields": ["prj_PRJRecNo", "prj_ModificationTS"],
                        "pk_guess": "prj_PRJRecNo",
                        "modification_ts_guess": "prj_ModificationTS",
                    }
                }
            }
        ),
    )
    _write_text(
        mapping_path,
        json.dumps(
            {
                "tables": {
                    "iSPEC_Projects": {
                        "local_table": "project",
                        "pk": {"legacy": "prj_PRJRecNo"},
                        "modified_ts": "prj_ModificationTS",
                    }
                }
            }
        ),
    )
    _write_text(tables_path, "tables\niSPEC_Projects\n")

    with get_session(file_path=str(core_db)) as session:
        session.add(
            Project(
                id=77,
                prj_AddedBy="tester",
                prj_ProjectTitle="Audit Project",
            )
        )

    with get_omics_session(file_path=str(core_db), logical_name="analysis"):
        pass
    with get_omics_session(file_path=str(core_db), logical_name="psm"):
        pass

    summary = write_import_audit_artifacts(
        db_file_path=str(core_db),
        analysis_db_file_path=str(core_db),
        psm_db_file_path=str(core_db),
        legacy_schema_path=str(schema_path),
        legacy_mapping_path=str(mapping_path),
        legacy_tables_file_path=str(tables_path),
        out_dir=str(out_dir),
    )

    audit_json = out_dir / "ispec-db-audit.json"
    audit_md = out_dir / "ispec-db-audit.md"
    gap_tsv = out_dir / "ispec-legacy-gap-matrix.tsv"

    assert summary["audit_json"] == str(audit_json)
    assert summary["audit_md"] == str(audit_md)
    assert summary["gap_tsv"] == str(gap_tsv)
    assert audit_json.exists()
    assert audit_md.exists()
    assert gap_tsv.exists()

    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    assert payload["core_database"]["exists"] is True
    assert payload["analysis_database"]["same_as_core"] is True
    assert payload["psm_database"]["same_as_core"] is True
    assert any(row["name"] == "project" for row in payload["core_database"]["tables"])
    assert any(row["name"] == "psm" for row in payload["psm_database"]["tables"])
    assert payload["legacy"]["gap_summary"]["mapped"] == 1
