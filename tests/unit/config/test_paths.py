from __future__ import annotations

from ispec.config.paths import (
    resolve_api_pid_file,
    resolve_db_location,
)
from ispec.config.audit import audit_environment


def test_resolve_analysis_db_defaults_to_sibling_of_core_db(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_ANALYSIS_DB_PATH", raising=False)
    monkeypatch.delenv("ISPEC_OMICS_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_DB_PATH", str(tmp_path / "core.db"))

    resolved = resolve_db_location("analysis")

    assert resolved.path == str(tmp_path / "ispec-analysis.db")
    assert resolved.source == "default_sibling"


def test_resolve_agent_state_db_defaults_to_sibling_of_core_db(tmp_path, monkeypatch):
    monkeypatch.delenv("ISPEC_AGENT_STATE_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_DB_PATH", str(tmp_path / "core.db"))

    resolved = resolve_db_location("agent_state")

    assert resolved.path == str(tmp_path / "ispec-agent-state.db")
    assert resolved.source == "default_sibling"


def test_resolve_analysis_db_uses_deprecated_alias_when_needed(tmp_path, monkeypatch):
    alias_path = tmp_path / "legacy-omics.db"
    monkeypatch.delenv("ISPEC_ANALYSIS_DB_PATH", raising=False)
    monkeypatch.setenv("ISPEC_OMICS_DB_PATH", str(alias_path))

    resolved = resolve_db_location("analysis")

    assert resolved.path == str(alias_path)
    assert resolved.source == "compat_env"
    assert resolved.deprecated_env_var == "ISPEC_OMICS_DB_PATH"


def test_resolve_api_pid_file_defaults_next_to_explicit_state_file(tmp_path, monkeypatch):
    state_path = tmp_path / "state" / "custom-api.json"
    monkeypatch.delenv("ISPEC_API_PID_FILE", raising=False)
    monkeypatch.setenv("ISPEC_API_STATE_FILE", str(state_path))

    resolved = resolve_api_pid_file()

    assert resolved.path == str(state_path.parent / "api_server.pid")
    assert resolved.source == "sibling_of_api_state_file"


def test_audit_environment_warns_for_deprecated_omics_alias(monkeypatch, tmp_path):
    monkeypatch.setenv("ISPEC_OMICS_DB_PATH", str(tmp_path / "legacy.db"))

    report = audit_environment(profile="dev")
    omics_var = next(item for item in report.vars if item.key == "ISPEC_OMICS_DB_PATH")

    assert any("Deprecated" in warning for warning in omics_var.warnings)
