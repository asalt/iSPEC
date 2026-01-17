import sys
from pathlib import Path

# Ensure the src directory is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ispec.cli.main import main


def test_cli_scaffold_import_script_writes_file(tmp_path, monkeypatch):
    results_dir = tmp_path / "Dec2025"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ispec",
            "db",
            "scaffold-import-script",
            "--project-id",
            "1544",
            "--results-dir",
            str(results_dir),
            "--out",
            str(out_dir),
            "--mspc-id",
            "MSPC001544",
            "--tag",
            "Dec2025",
            "--with-gct-export",
        ],
    )
    main()

    script_path = out_dir / "import_project1544_mspc001544_dec2025.sh"
    assert script_path.exists()

    content = script_path.read_text(encoding="utf-8")
    assert 'PROJECT_ID="${PROJECT_ID:-1544}"' in content
    assert str(results_dir) in content
    assert "ISPEC_ENV_FILE" in content
    assert "db import-results" in content
    assert "== Import: GCT exports ==" in content

