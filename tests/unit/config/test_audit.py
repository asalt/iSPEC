from ispec.config.audit import audit_environment, init_env_files


def _find(report, key: str):
    for item in report.vars:
        if item.key == key:
            return item
    raise AssertionError(f"Missing audit entry for {key}")


def test_audit_environment_flags_prod_missing_secrets():
    env = {
        "ISPEC_REQUIRE_LOGIN": "1",
        "ISPEC_DEV_DEFAULT_ADMIN": "0",
        "ISPEC_ASSISTANT_PROVIDER": "stub",
    }
    report = audit_environment(env, profile="prod")
    assert report.ok is False
    assert _find(report, "ISPEC_API_KEY").errors
    assert _find(report, "ISPEC_PASSWORD_PEPPER").errors


def test_audit_environment_flags_dev_admin_enabled_in_prod():
    env = {
        "ISPEC_API_KEY": "not-a-placeholder-but-set",
        "ISPEC_PASSWORD_PEPPER": "also-set-and-long-enough",
        "ISPEC_REQUIRE_LOGIN": "1",
        "ISPEC_DEV_DEFAULT_ADMIN": "1",
        "ISPEC_ASSISTANT_PROVIDER": "stub",
    }
    report = audit_environment(env, profile="prod")
    assert report.ok is False
    assert any("production" in msg.lower() for msg in _find(report, "ISPEC_DEV_DEFAULT_ADMIN").errors)


def test_init_env_files_noninteractive_generates_required_secrets():
    base, assistant = init_env_files(profile="prod", interactive=False)
    assert base["ISPEC_API_KEY"]
    assert base["ISPEC_PASSWORD_PEPPER"]
    assert base.get("ISPEC_DEV_DEFAULT_ADMIN") == "0"
    assert assistant["ISPEC_ASSISTANT_PROVIDER"] == "vllm"
    assert assistant["ISPEC_VLLM_URL"].startswith("http")

