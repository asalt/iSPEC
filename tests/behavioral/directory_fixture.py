"""Evaluation-only insertion points in the gate's private application copy."""

from pathlib import Path


def labeled_directory(directory, exposed):
    """Describe which authorized groups already have all their schemas loaded."""
    from scripts.jeff_router_catalog import DIRECTORY_LABELS

    opened = [name for name, tools in directory.groups.items() if tools <= exposed]
    remaining = [f"{name}: {DIRECTORY_LABELS[name]}" for name in directory.groups if name not in opened]
    schema = directory.directory_schema()
    schema["function"]["description"] = (
        "Tools already available: " + (", ".join(opened) or "none") +
        ". Call those tools directly. To use another category, open it first.\n"
        "Categories available to open:\n" + ("\n".join(remaining) or "none")
    )
    return schema


def adapt_snapshot(route: Path):
    if "source" not in route.parents[4].parts and "source" not in route.parts:
        raise ValueError("Expected private gate snapshot")
    source = route.read_text()
    anchor = "    tool_schemas_count = len(tool_schemas)"
    position = source.index(anchor, source.index("# Routing may select read tools"))
    source = source[:position] + '''    from tests.behavioral.test_directory_live import turn_directory
    from tests.behavioral.directory_fixture import labeled_directory
    directory, initial_groups = turn_directory(user)
    exposed = {name for group in initial_groups for name in directory.groups[group]}
    tool_schemas = [labeled_directory(directory, exposed), *(directory.schemas[name] for name in sorted(exposed))]
''' + source[position:]
    anchor = "                elif unavailable_tool_payload is not None:"
    assert source.count(anchor) == 1
    source = source.replace(anchor, '''                elif tool_name == "open_tool_category":
                    used_tool_calls += 1
                    try:
                        revealed = directory.open_category(name=parsed_args.get("name"))
                        tool_schemas.extend(spec for spec in revealed if spec["function"]["name"] not in openai_tool_names)
                        tool_schemas[0] = labeled_directory(directory, _openai_tool_names(tool_schemas))
                        _refresh_tool_schema_state()
                        tool_payload = {"ok": True, "tools": [s["function"]["name"] for s in revealed]}
                    except ValueError:
                        tool_payload = {"ok": False, "error": "Category unavailable to this user."}
''' + anchor)
    route.write_text(source)
