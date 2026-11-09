"""Pure, evaluation-only layout helpers. No production prompt/policy changes."""

from copy import deepcopy


def merge_adjacent_system_messages(messages):
    """Preserve ordered text and roles; never elevate user/tool evidence."""
    result = []
    for message in deepcopy(messages):
        if (
            result
            and message.get("role") == result[-1].get("role") == "system"
            and set(message) == set(result[-1]) == {"role", "content"}
            and isinstance(message["content"], str)
            and isinstance(result[-1]["content"], str)
        ):
            result[-1]["content"] += "\n\n" + message["content"]
        else:
            result.append(message)
    return result


def layout_summary(messages, *, base_text, current_request):
    contents = [str(message.get("content") or "") for message in messages]
    return {
        "roles": [message["role"] for message in messages],
        "content_chars": sum(map(len, contents)),
        "system_chars": sum(
            len(text)
            for message, text in zip(messages, contents)
            if message["role"] == "system"
        ),
        "base_occurrences": sum(text.count(base_text) for text in contents),
        "current_request_occurrences": sum(
            text.count(current_request) for text in contents
        ),
    }
