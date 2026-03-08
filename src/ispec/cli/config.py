from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from ispec.config.audit import audit_environment, init_env_files, render_env_file
from ispec.config.contract import default_contract, spec_to_dict
from ispec.config.paths import resolved_path_catalog
from ispec.cli.env import parse_env_file_text


def _profile(value: str) -> Profile:
    lowered = value.strip().lower()
    if lowered in {"dev", "prod"}:
        return lowered  # type: ignore[return-value]
    raise SystemExit(f"Unknown profile: {value}")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return parse_env_file_text(path.read_text(encoding="utf-8"))


def register_subcommands(subparsers):
    audit = subparsers.add_parser("audit", help="Audit current environment against the config contract.")
    audit.add_argument("--profile", default="dev", choices=["dev", "prod"])
    audit.add_argument("--format", default="human", choices=["human", "json"])
    audit.add_argument("--reveal-secrets", action="store_true", help="Include secret values in JSON output.")

    init = subparsers.add_parser("init", help="Interactive env file initializer (prompts + secret generation).")
    init.add_argument("--profile", default="dev", choices=["dev", "prod"])
    init.add_argument(
        "--write-env-file",
        default=".env.local",
        help="Env file to write (base). Defaults to .env.local in the current directory.",
    )
    init.add_argument(
        "--write-assistant-env-file",
        default=".env.vllm.local",
        help="Env file to write assistant settings into (optional overlay). Defaults to .env.vllm.local.",
    )
    init.add_argument(
        "--single-file",
        action="store_true",
        help="Write everything into --write-env-file (do not split assistant settings).",
    )
    init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt; fill defaults and generate required secrets when possible.",
    )
    init.add_argument("--stdout", action="store_true", help="Print rendered env file(s) instead of writing.")

    contract = subparsers.add_parser("contract", help="Print the config contract.")
    contract.add_argument("--format", default="json", choices=["json"])

    paths = subparsers.add_parser("paths", help="Show resolved storage/log/state paths.")
    paths.add_argument("--format", default="human", choices=["human", "json"])


def _human_print_report(report, *, console: Console) -> None:
    console.print(f"Profile: [bold]{report.profile}[/bold]")
    console.print(f"OK: {report.ok}  errors={report.errors} warnings={report.warnings}")
    if report.ok and report.warnings == 0:
        return
    console.print("")
    for item in report.vars:
        if not item.errors and not item.warnings:
            continue
        status = "ERROR" if item.errors else "WARN"
        console.print(f"[{status}] {item.key} ({item.group})")
        for err in item.errors:
            console.print(f"  - {err}")
        for warn in item.warnings:
            console.print(f"  - {warn}")


def _human_print_paths(*, console: Console) -> None:
    catalog = resolved_path_catalog()
    for group_name, items in catalog.items():
        console.print(f"[bold]{group_name.title()}[/bold]")
        for _, resolved in items.items():
            line = f"- {resolved.name}: {resolved.value}"
            details: list[str] = [resolved.source]
            if resolved.env_var:
                details.append(resolved.env_var)
            if resolved.deprecated_env_var:
                details.append(f"deprecated={resolved.deprecated_env_var}")
            if resolved.uri and resolved.uri != resolved.value:
                details.append(f"uri={resolved.uri}")
            console.print(f"{line} [{' ; '.join(details)}]")
            for note in resolved.notes:
                console.print(f"  note: {note}")
        console.print("")


def dispatch(args) -> None:
    console = Console()
    if args.subcommand == "contract":
        payload = [spec_to_dict(spec) for spec in default_contract()]
        console.print_json(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return

    if args.subcommand == "paths":
        payload = {
            group: {name: resolved.as_dict() for name, resolved in items.items()}
            for group, items in resolved_path_catalog().items()
        }
        if args.format == "json":
            console.print_json(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            _human_print_paths(console=console)
        return

    if args.subcommand == "audit":
        report = audit_environment(profile=_profile(args.profile))
        if args.format == "json":
            console.print_json(report.to_json(reveal_secrets=bool(args.reveal_secrets)))
        else:
            _human_print_report(report, console=console)
        if not report.ok:
            raise SystemExit(2)
        return

    if args.subcommand == "init":
        profile = _profile(args.profile)
        base_path = Path(args.write_env_file).expanduser()
        assistant_path = Path(args.write_assistant_env_file).expanduser()
        base_existing = _read_env_file(base_path)
        assistant_existing = {} if args.single_file else _read_env_file(assistant_path)

        base_values, assistant_values = init_env_files(
            profile=profile,
            base_values=base_existing,
            assistant_values={} if args.single_file else assistant_existing,
            interactive=not bool(args.non_interactive),
        )

        header = "Generated by `ispec config init`. Edit as needed."
        if args.single_file:
            merged = {**base_values, **assistant_values}
            rendered = render_env_file(merged, header=header)
            if args.stdout:
                console.print(rendered, end="")
            else:
                base_path.write_text(rendered, encoding="utf-8")
                console.print(f"Wrote {base_path}")
            return

        rendered_base = render_env_file(base_values, header=header)
        rendered_assistant = render_env_file(assistant_values, header=header)
        if args.stdout:
            console.print(rendered_base, end="")
            console.print(rendered_assistant, end="")
        else:
            base_path.write_text(rendered_base, encoding="utf-8")
            assistant_path.write_text(rendered_assistant, encoding="utf-8")
            console.print(f"Wrote {base_path}")
            console.print(f"Wrote {assistant_path}")
        return

    raise SystemExit(f"Unknown config subcommand: {args.subcommand}")
