from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import secrets
from typing import Any

from ispec.config.contract import GeneratorSpec, Profile, VarSpec, default_contract


_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _normalize_raw(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _default_for(spec: VarSpec, profile: Profile) -> str | None:
    return spec.default_for(profile)


def _required(spec: VarSpec, env: dict[str, str], profile: Profile) -> bool:
    if profile in spec.required_in:
        return True
    if spec.required_if is None:
        return False
    current = (env.get(spec.required_if.key) or "").strip()
    return current == spec.required_if.equals


def _validate(spec: VarSpec, raw: str | None) -> list[str]:
    errors: list[str] = []
    value = _normalize_raw(raw)
    if value is None:
        return errors

    if spec.forbid_values and value in set(spec.forbid_values):
        errors.append(f"Value is a placeholder/insecure default ({value!r}).")

    if spec.kind == "bool":
        lower = value.lower()
        if lower not in _TRUTHY and lower not in _FALSY:
            errors.append("Must be a boolean-ish value (1/0/true/false/yes/no).")
        return errors

    if spec.kind == "int":
        try:
            parsed = int(value)
        except ValueError:
            errors.append("Must be an integer.")
            return errors
        if spec.min_value is not None and parsed < spec.min_value:
            errors.append(f"Must be >= {spec.min_value}.")
        if spec.max_value is not None and parsed > spec.max_value:
            errors.append(f"Must be <= {spec.max_value}.")
        return errors

    if spec.min_length is not None and len(value) < spec.min_length:
        errors.append(f"Must be at least {spec.min_length} characters.")

    if spec.choices and value not in set(spec.choices):
        errors.append(f"Must be one of: {', '.join(spec.choices)}.")

    if spec.kind == "url":
        if not (value.startswith("http://") or value.startswith("https://")):
            errors.append("Must start with http:// or https://")

    if spec.kind == "path":
        if "://" in value and not value.startswith("sqlite"):
            errors.append("Expected a filesystem path or sqlite:/// URI.")

    return errors


@dataclass(frozen=True)
class VarAudit:
    key: str
    group: str
    required: bool
    present: bool
    value: str | None
    default: str | None
    errors: list[str]
    warnings: list[str]
    secret: bool = False

    def redacted_value(self) -> str | None:
        if not self.secret:
            return self.value
        if self.value is None:
            return None
        return "<set>"


@dataclass(frozen=True)
class AuditReport:
    profile: Profile
    ok: bool
    errors: int
    warnings: int
    vars: list[VarAudit]

    def to_json(self, *, reveal_secrets: bool = False) -> str:
        payload: dict[str, Any] = {
            "profile": self.profile,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "vars": [],
        }
        for item in self.vars:
            entry = asdict(item)
            if item.secret and not reveal_secrets:
                entry["value"] = item.redacted_value()
            payload["vars"].append(entry)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def audit_environment(
    env: dict[str, str] | None = None,
    *,
    profile: Profile = "dev",
    contract: tuple[VarSpec, ...] | None = None,
) -> AuditReport:
    if env is None:
        import os

        env = dict(os.environ)

    contract = contract or default_contract()
    items: list[VarAudit] = []
    error_count = 0
    warning_count = 0

    for spec in contract:
        raw = env.get(spec.key)
        default = _default_for(spec, profile)
        required = _required(spec, env, profile)
        present = _normalize_raw(raw) is not None

        errs = _validate(spec, raw)
        warns: list[str] = []

        if required and not present:
            errs = [*errs, "Missing required value."]

        if not required and not present:
            if profile in spec.recommended_in:
                warns.append("Missing recommended value.")

        if spec.kind == "path" and present:
            val = str(raw).strip()
            if not val.startswith("sqlite") and not Path(val).is_absolute():
                warns.append("Relative path; prefer absolute in production.")

        if spec.key == "ISPEC_DEV_DEFAULT_ADMIN" and profile == "prod":
            if present and _is_truthy(raw):
                errs.append("Dev default admin must be disabled in production.")

        if errs:
            error_count += 1
        if warns:
            warning_count += 1

        items.append(
            VarAudit(
                key=spec.key,
                group=spec.group,
                required=required,
                present=present,
                value=_normalize_raw(raw),
                default=default,
                errors=errs,
                warnings=warns,
                secret=spec.secret,
            )
        )

    return AuditReport(
        profile=profile,
        ok=error_count == 0,
        errors=error_count,
        warnings=warning_count,
        vars=items,
    )


def generate_secret(spec: GeneratorSpec) -> str:
    if spec.kind == "token_urlsafe":
        return secrets.token_urlsafe(spec.nbytes)
    raise ValueError(f"Unknown generator kind: {spec.kind}")


def _prompt(text: str) -> str:
    return input(text)


def _coerce_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    lower = raw.strip().lower()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    return default


def _format_env_value(value: str) -> str:
    if not value:
        return ""
    if any(ch.isspace() for ch in value) or "#" in value:
        escaped = value.replace('"', '\\"')
        return f"\"{escaped}\""
    return value


def render_env_file(
    values: dict[str, str],
    *,
    contract: tuple[VarSpec, ...] | None = None,
    header: str | None = None,
) -> str:
    contract = contract or default_contract()
    known = {spec.key for spec in contract}

    groups: dict[str, list[str]] = {}
    for spec in contract:
        groups.setdefault(spec.group, []).append(spec.key)

    out: list[str] = []
    if header:
        out.extend([f"# {line}".rstrip() for line in header.splitlines()])
        out.append("")

    def emit_group(group: str, keys: list[str]) -> None:
        emitted = False
        for key in keys:
            if key not in values:
                continue
            val = values.get(key)
            if val is None:
                continue
            if not emitted:
                out.append(f"# {group}")
                emitted = True
            out.append(f"{key}={_format_env_value(str(val))}")
        if emitted:
            out.append("")

    for group, keys in groups.items():
        emit_group(group, keys)

    extra_keys = sorted(k for k in values.keys() if k not in known)
    if extra_keys:
        emit_group("Other", extra_keys)

    while out and out[-1] == "":
        out.pop()
    out.append("")
    return "\n".join(out)


def init_env_files(
    *,
    profile: Profile,
    base_values: dict[str, str] | None = None,
    assistant_values: dict[str, str] | None = None,
    contract: tuple[VarSpec, ...] | None = None,
    interactive: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    contract = contract or default_contract()
    base_values = dict(base_values or {})
    assistant_values = dict(assistant_values or {})

    def target_dict(spec: VarSpec) -> dict[str, str]:
        if spec.group == "Assistant":
            return assistant_values
        return base_values

    for spec in contract:
        env_view = {**base_values, **assistant_values}
        store = target_dict(spec)
        current = _normalize_raw(store.get(spec.key))
        default = _default_for(spec, profile)
        required = _required(spec, env_view, profile)

        if current is None and default is not None:
            current = default

        if not interactive:
            if current is None:
                if required and spec.generator is not None:
                    store[spec.key] = generate_secret(spec.generator)
                elif required:
                    raise SystemExit(f"Missing required {spec.key} (no default/generator).")
                continue
            store[spec.key] = current
            continue

        prompt_label = f"{spec.key} [{spec.kind}]"
        if required:
            prompt_label += " (required)"

        if spec.secret:
            shown = "<set>" if current else "<empty>"
        else:
            shown = current or "<empty>"

        print(f"\n{prompt_label}\n  {spec.description}\n  current/default: {shown}")

        if spec.kind == "bool":
            default_bool = _coerce_bool(current, default=False)
            raw = _prompt(f"  Set to true? [{'Y/n' if default_bool else 'y/N'}]: ").strip()
            if not raw:
                store[spec.key] = "1" if default_bool else "0"
            else:
                store[spec.key] = "1" if raw.lower() in _TRUTHY else "0"
            continue

        if spec.kind == "int":
            raw = _prompt(f"  Value [{current or ''}]: ").strip()
            store[spec.key] = raw or (current or "")
            continue

        if spec.choices:
            choice_line = ", ".join(spec.choices)
            raw = _prompt(f"  Value ({choice_line}) [{current or ''}]: ").strip()
            store[spec.key] = raw or (current or "")
            continue

        if spec.secret and spec.generator is not None and (current is None or current in spec.forbid_values):
            raw = _prompt("  Generate a new secret? [Y/n]: ").strip().lower()
            if not raw or raw in _TRUTHY:
                store[spec.key] = generate_secret(spec.generator)
                continue

        raw = _prompt(f"  Value [{current or ''}]: ")
        store[spec.key] = raw.strip() or (current or "")

    return base_values, assistant_values
