"""Edit a local registry export or derive initial monitoring entries from fish profiles."""

import argparse
import json
from pathlib import Path
import subprocess

from ispec.db.connect import get_session
from ispec.db.models import InstrumentRegistry
from ispec.instruments.registry import Entry, add_entry, change_entry, record


def profile(name):
    # Trusted installed wrappers; use their read-only operations, discard login fields.
    script = """source "$__fish_config_dir/functions/__mount_smb_profile.fish"; or exit 1
    source "$__fish_config_dir/functions/__mount_host_ip.fish"; or exit 1
    set p (string split \\t -- (__mount_smb_profile $argv[1])); or exit 1
    set ip (__mount_host_ip $p[2]); or exit 1
    printf '%s\\t%s\\t%s\\n' $ip $p[3] $p[4]"""
    data = subprocess.check_output(
        ["fish", "--no-config", "-c", script, name], text=True, timeout=5
    )
    host, share, target = data.rstrip("\n").split("\t")
    return dict(host=host, share=share, mount_path=target)


def fish_entries():
    ecl, data = profile("eclipse_c"), profile("eclipse_d")
    c, d = profile("timstof_c"), profile("timstof_dshare")
    system = "Thermo Scientific/Instruments/TNG/OrbitrapEclipse/4.3/System"
    for key, name, plugin, sources in [
        (
            "eclipse",
            "Orbitrap Eclipse",
            "thermo",
            {
                "acquisition": (data, "QC"),
                "log": (ecl, "ProgramData/" + system + "/logs"),
                "tune": (ecl, "Program Files/" + system + "/MSI"),
            },
        ),
        (
            "tims-ultra-0006",
            "timsTOF Ultra",
            "bruker",
            {
                "acquisition": (d, "58799_58810"),
                "log": (c, "BDalSystemData/HyStar/LogFiles"),
            },
        ),
    ]:
        yield Entry(
            id=key,
            display_name=name,
            plugin=plugin,
            mode="pull",
            roots={
                k: str(Path(m["mount_path"]) / suffix)
                for k, (m, suffix) in sources.items()
            },
            connections={k: m for k, (m, _) in sources.items()},
        ).model_dump()


def apply(db, entries):
    for raw in entries:
        raw = dict(raw)
        revision = raw.pop("revision", None)
        entry = Entry.model_validate(raw)
        row = db.get(InstrumentRegistry, entry.id)
        if row is None and revision is None:
            add_entry(db, entry)
        elif row and row.revision == revision:
            change_entry(db, row, entry.model_dump(exclude={"id"}))
        else:
            raise ValueError(f"{entry.id}: stale export or existing ID; export again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fish-preview")
    commands.add_parser("export")
    commands.add_parser("apply").add_argument("file", type=Path)
    args = parser.parse_args()
    if args.command == "fish-preview":
        print(json.dumps(list(fish_entries()), indent=2))
        return
    with get_session(args.database) as db:
        if args.command == "export":
            print(
                json.dumps(
                    [
                        record(r)
                        for r in db.query(InstrumentRegistry).order_by(
                            InstrumentRegistry.id
                        )
                    ],
                    indent=2,
                )
            )
        else:
            apply(db, json.loads(args.file.read_text()))


if __name__ == "__main__":
    main()
