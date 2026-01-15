"""Command-line helpers for managing iSPEC authentication users.

This module focuses on provisioning user accounts with temporary passwords that
must be changed on first login (``must_change_password=True``).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import secrets
from typing import Iterable, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ispec.logging import get_logger

logger = get_logger(__name__)

_ROLE_CHOICES = ("admin", "editor", "viewer", "client")

_PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"


@dataclass(frozen=True)
class ProvisionedCredential:
    user_id: int
    username: str
    password: str
    role: str
    action: Literal["created", "reset"]
    project_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CredentialCheck:
    username: str
    found: bool
    is_active: bool | None
    role: str | None
    must_change_password: bool | None
    password_ok: bool
    reason: str


def _dedupe_usernames(usernames: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in usernames:
        username = (raw or "").strip()
        if not username:
            continue
        if username in seen:
            continue
        seen.add(username)
        deduped.append(username)
    return deduped


def _generate_password(length: int) -> str:
    if length < 8:
        raise ValueError("password length must be at least 8 characters")
    return "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(length))


def _infer_output_format(path: Path) -> Literal["csv", "tsv", "json"]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    return "tsv"


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Best-effort: Windows/filesystems may not support POSIX modes.
        pass


def _write_credentials(path: Path, credentials: list[ProvisionedCredential]) -> Literal["csv", "tsv", "json"]:
    fmt = _infer_output_format(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        payload = [
            {
                "user_id": cred.user_id,
                "username": cred.username,
                "password": cred.password,
                "role": cred.role,
                "action": cred.action,
            }
            for cred in credentials
        ]
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _chmod_private(path)
        return fmt

    delimiter = "," if fmt == "csv" else "\t"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["username", "password", "role", "action"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for cred in credentials:
            writer.writerow(
                {
                    "username": cred.username,
                    "password": cred.password,
                    "role": cred.role,
                    "action": cred.action,
                }
            )
    _chmod_private(path)
    return fmt


def _render_summary(
    credentials: list[ProvisionedCredential],
    *,
    database: str | None,
    output: Path,
    output_format: str,
    print_passwords: bool,
    console: Console | None = None,
) -> None:
    if console is None:
        console = Console()

    show_project_ids = any(cred.project_ids is not None for cred in credentials)

    created = sum(1 for c in credentials if c.action == "created")
    reset = sum(1 for c in credentials if c.action == "reset")
    console.print(
        f"Provisioned {len(credentials)} user(s): {created} created, {reset} reset.",
        highlight=False,
    )
    if database:
        console.print(f"Database: {database}", highlight=False)
    else:
        console.print("Database: (default via ISPEC_DB_PATH/ISPEC_DB_DIR)", highlight=False)

    pepper_raw = (
        os.getenv("ISPEC_PASSWORD_PEPPER")
        or os.getenv("ISPEC_AUTH_PEPPER")
        or os.getenv("ISPEC_PASSWORD_SALT")
        or ""
    )
    pepper_state = "set" if pepper_raw.strip() else "unset"
    console.print(
        f"Password pepper: {pepper_state} (must match the API server environment)",
        highlight=False,
    )
    console.print(f"Wrote credentials to: {output} ({output_format})", highlight=False)

    table = Table(title="Provisioned users")
    table.add_column("username", style="bold")
    if print_passwords:
        table.add_column("password")
    table.add_column("role")
    table.add_column("action")
    if show_project_ids:
        table.add_column("project_ids")
    for cred in credentials:
        row = [cred.username]
        if print_passwords:
            row.append(cred.password)
        row.extend([cred.role, cred.action])
        if show_project_ids:
            if cred.project_ids is None:
                row.append("")
            elif not cred.project_ids:
                row.append("(none)")
            else:
                row.append(",".join(str(pid) for pid in cred.project_ids))
        table.add_row(*row)
    console.print(table)

    if not pepper_raw.strip():
        console.print(
            Panel.fit(
                "Tip: if you use `.env.local`, its KEY=value lines are not exported by default.\n"
                "Run `set -a; source .env.local; set +a`, pass `--password-pepper ...`,\n"
                "or use `ispec --env-file .env.local ...`.",
                title="Pepper note",
            )
        )


def _apply_password_pepper_override(password_pepper: str | None) -> None:
    if password_pepper is None:
        return
    value = password_pepper.strip()
    if not value:
        return
    os.environ["ISPEC_PASSWORD_PEPPER"] = value


def check_credentials(
    *,
    username: str,
    password: str,
    database: str | None,
) -> CredentialCheck:
    from ispec.api.security import verify_password
    from ispec.db.connect import get_session
    from ispec.db.models import AuthUser

    normalized = (username or "").strip()
    if not normalized:
        raise SystemExit("Username is required.")

    with get_session(database) as db:
        user = db.query(AuthUser).filter(AuthUser.username == normalized).first()
        if user is None:
            return CredentialCheck(
                username=normalized,
                found=False,
                is_active=None,
                role=None,
                must_change_password=None,
                password_ok=False,
                reason="user not found",
            )

        is_active = bool(user.is_active)
        role = str(user.role.value if hasattr(user.role, "value") else user.role)
        must_change = bool(getattr(user, "must_change_password", False))
        if not is_active:
            return CredentialCheck(
                username=normalized,
                found=True,
                is_active=is_active,
                role=role,
                must_change_password=must_change,
                password_ok=False,
                reason="user is inactive",
            )

        ok = verify_password(
            password,
            salt_b64=user.password_salt,
            hash_b64=user.password_hash,
            iterations=user.password_iterations,
        )
        return CredentialCheck(
            username=normalized,
            found=True,
            is_active=is_active,
            role=role,
            must_change_password=must_change,
            password_ok=ok,
            reason="ok" if ok else "invalid password",
        )


def _render_check_result(
    result: CredentialCheck,
    *,
    database: str | None,
    console: Console | None = None,
) -> None:
    if console is None:
        console = Console()

    pepper_raw = (
        os.getenv("ISPEC_PASSWORD_PEPPER")
        or os.getenv("ISPEC_AUTH_PEPPER")
        or os.getenv("ISPEC_PASSWORD_SALT")
        or ""
    )
    pepper_state = "set" if pepper_raw.strip() else "unset"

    if database:
        console.print(f"Database: {database}", highlight=False)
    else:
        console.print("Database: (default via ISPEC_DB_PATH/ISPEC_DB_DIR)", highlight=False)
    console.print(f"Password pepper: {pepper_state}", highlight=False)

    table = Table(title="Credential check")
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("username", result.username)
    table.add_row("found", str(result.found))
    if result.found:
        table.add_row("is_active", str(result.is_active))
        table.add_row("role", str(result.role))
        table.add_row("must_change_password", str(result.must_change_password))
    table.add_row("password_ok", str(result.password_ok))
    table.add_row("reason", result.reason)
    console.print(table)


def provision_users(
    usernames: Iterable[str],
    *,
    database: str | None,
    role: str,
    reset_existing: bool,
    password_length: int,
    activate: bool,
    update_role: bool,
    project_ids: Iterable[int] | None = None,
    experiment_ids: Iterable[int] | None = None,
    clear_project_access: bool = False,
) -> list[ProvisionedCredential]:
    """Create or reset users with generated passwords and forced password change."""

    from ispec.api.security import hash_password
    from ispec.db.connect import get_session
    from ispec.db.models import AuthSession, AuthUser, AuthUserProject, Experiment, Project, UserRole

    try:
        role_enum = UserRole(role)
    except ValueError as exc:
        raise SystemExit(f"Unknown role {role!r}. Expected one of: {', '.join(_ROLE_CHOICES)}") from exc

    normalized = _dedupe_usernames(usernames)
    if not normalized:
        raise SystemExit("No usernames provided.")

    now = datetime.now(UTC)
    results: list[ProvisionedCredential] = []

    with get_session(database) as db:
        selected_project_ids: list[int] = []
        requested_projects = [int(pid) for pid in (project_ids or [])]
        requested_experiments = [int(eid) for eid in (experiment_ids or [])]

        def dedupe_ints(values: list[int]) -> list[int]:
            seen: set[int] = set()
            deduped: list[int] = []
            for value in values:
                try:
                    value_int = int(value)
                except Exception:
                    continue
                if value_int <= 0 or value_int in seen:
                    continue
                seen.add(value_int)
                deduped.append(value_int)
            return deduped

        requested_projects = dedupe_ints(requested_projects)
        requested_experiments = dedupe_ints(requested_experiments)

        update_project_access = bool(clear_project_access or requested_projects or requested_experiments)
        if update_project_access:
            if requested_projects:
                existing_projects = {
                    int(row[0]) for row in db.query(Project.id).filter(Project.id.in_(requested_projects)).all()
                }
                missing = [pid for pid in requested_projects if pid not in existing_projects]
                if missing:
                    raise SystemExit(f"Unknown project ids: {', '.join(map(str, missing))}")
                selected_project_ids.extend(pid for pid in requested_projects if pid in existing_projects)

            if requested_experiments:
                rows = (
                    db.query(Experiment.id, Experiment.project_id)
                    .filter(Experiment.id.in_(requested_experiments))
                    .all()
                )
                found = {int(row[0]) for row in rows}
                missing = [eid for eid in requested_experiments if eid not in found]
                if missing:
                    raise SystemExit(f"Unknown experiment ids: {', '.join(map(str, missing))}")
                no_project = [int(row[0]) for row in rows if row[1] is None]
                if no_project:
                    raise SystemExit(
                        "Experiment(s) missing project_id: " + ", ".join(map(str, sorted(no_project)))
                    )
                selected_project_ids.extend(int(row[1]) for row in rows if row[1] is not None)

            selected_project_ids = sorted(set(dedupe_ints(selected_project_ids)))

        for username in normalized:
            existing = db.query(AuthUser).filter(AuthUser.username == username).first()

            password = _generate_password(password_length)
            salt_b64, hash_b64, iterations = hash_password(password)

            if existing is not None:
                if not reset_existing:
                    raise SystemExit(
                        f"User {username!r} already exists; re-run with --reset-existing to reset password."
                    )

                existing.password_hash = hash_b64
                existing.password_salt = salt_b64
                existing.password_iterations = iterations
                existing.must_change_password = True
                existing.password_changed_at = now
                if activate:
                    existing.is_active = True
                if update_role:
                    existing.role = role_enum

                db.query(AuthSession).filter(AuthSession.user_id == existing.id).delete()
                if update_project_access:
                    db.query(AuthUserProject).filter(AuthUserProject.user_id == existing.id).delete()
                    for project_id in selected_project_ids:
                        db.add(AuthUserProject(user_id=int(existing.id), project_id=int(project_id)))
                results.append(
                    ProvisionedCredential(
                        user_id=int(existing.id),
                        username=existing.username,
                        password=password,
                        role=str(existing.role.value if hasattr(existing.role, "value") else existing.role),
                        action="reset",
                        project_ids=tuple(selected_project_ids) if update_project_access else None,
                    )
                )
                continue

            user = AuthUser(
                username=username,
                password_hash=hash_b64,
                password_salt=salt_b64,
                password_iterations=iterations,
                role=role_enum,
                is_active=bool(activate),
                must_change_password=True,
                password_changed_at=now,
            )
            db.add(user)
            db.flush()
            if update_project_access:
                for project_id in selected_project_ids:
                    db.add(AuthUserProject(user_id=int(user.id), project_id=int(project_id)))
            results.append(
                ProvisionedCredential(
                    user_id=int(user.id),
                    username=user.username,
                    password=password,
                    role=str(user.role.value if hasattr(user.role, "value") else user.role),
                    action="created",
                    project_ids=tuple(selected_project_ids) if update_project_access else None,
                )
            )

    return results


def register_subcommands(subparsers) -> None:
    provision_parser = subparsers.add_parser(
        "provision",
        help="Provision users with temporary passwords (forces change on first login).",
    )
    provision_parser.add_argument(
        "usernames",
        nargs="+",
        help="Usernames to create/reset (space separated).",
    )
    provision_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path (defaults to ISPEC_DB_PATH/default).",
    )
    provision_parser.add_argument(
        "--role",
        default="editor",
        choices=_ROLE_CHOICES,
        help="Role to assign (default: editor).",
    )
    provision_parser.add_argument(
        "--reset-existing",
        action="store_true",
        help="Reset passwords for existing users instead of erroring.",
    )
    provision_parser.add_argument(
        "--password-pepper",
        dest="password_pepper",
        help="Override ISPEC_PASSWORD_PEPPER for this run (must match the API server).",
    )
    provision_parser.add_argument(
        "--password-length",
        dest="password_length",
        type=int,
        default=16,
        help="Generated password length (default: 16; min: 8).",
    )
    provision_parser.add_argument(
        "--output",
        required=True,
        help="Credential output file (.csv, .tsv, or .json).",
    )
    provision_parser.add_argument(
        "--print-passwords",
        action="store_true",
        help="Also print passwords to stdout (use with caution).",
    )
    provision_parser.add_argument(
        "--no-activate",
        dest="activate",
        action="store_false",
        default=True,
        help="Do not force is_active=True for provisioned accounts.",
    )
    provision_parser.add_argument(
        "--no-role-update",
        dest="update_role",
        action="store_false",
        default=True,
        help="When resetting existing users, keep their current role.",
    )
    provision_parser.add_argument(
        "--project-id",
        dest="project_ids",
        action="append",
        type=int,
        default=[],
        help="Grant access to a project id (repeatable). For client accounts, this limits visible projects.",
    )
    provision_parser.add_argument(
        "--experiment-id",
        dest="experiment_ids",
        action="append",
        type=int,
        default=[],
        help="Grant access based on experiment id(s) (repeatable). Adds the parent project(s) for these experiments.",
    )
    provision_parser.add_argument(
        "--clear-project-access",
        dest="clear_project_access",
        action="store_true",
        help="Clear auth_user_project entries for the provisioned user(s) (client accounts will see no projects).",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Check whether a username/password matches the DB (uses current pepper env).",
    )
    check_parser.add_argument("username", help="Username to check.")
    check_parser.add_argument("password", help="Password to verify.")
    check_parser.add_argument(
        "--database",
        dest="database",
        help="SQLite database URL or filesystem path (defaults to ISPEC_DB_PATH/default).",
    )
    check_parser.add_argument(
        "--password-pepper",
        dest="password_pepper",
        help="Override ISPEC_PASSWORD_PEPPER for this check.",
    )


def dispatch(args) -> None:
    if args.subcommand == "provision":
        output = Path(args.output)
        database = getattr(args, "database", None)
        _apply_password_pepper_override(getattr(args, "password_pepper", None))
        credentials = provision_users(
            args.usernames,
            database=database,
            role=str(args.role),
            reset_existing=bool(args.reset_existing),
            password_length=int(args.password_length),
            activate=bool(args.activate),
            update_role=bool(args.update_role),
            project_ids=getattr(args, "project_ids", None),
            experiment_ids=getattr(args, "experiment_ids", None),
            clear_project_access=bool(getattr(args, "clear_project_access", False)),
        )
        output_format = _write_credentials(output, credentials)
        _render_summary(
            credentials,
            database=database,
            output=output,
            output_format=output_format,
            print_passwords=bool(args.print_passwords),
        )
        return

    if args.subcommand == "check":
        database = getattr(args, "database", None)
        _apply_password_pepper_override(getattr(args, "password_pepper", None))
        result = check_credentials(
            username=str(args.username),
            password=str(args.password),
            database=database,
        )
        _render_check_result(result, database=database)
        if not result.password_ok:
            raise SystemExit(1)
        return

    logger.error("No handler for auth subcommand: %s", getattr(args, "subcommand", None))
    raise SystemExit(2)
