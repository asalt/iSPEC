import os

import pytest

from ispec.cli.env import extract_env_files, load_env_files, parse_env_file_text


def test_extract_env_files_supports_both_forms():
    env_files, argv = extract_env_files(
        [
            "auth",
            "provision",
            "antrixj",
            "--output",
            "out.csv",
            "--env-file",
            ".env.local",
            "--password-length",
            "12",
            "--env-file=extra.env",
        ]
    )
    assert env_files == [".env.local", "extra.env"]
    assert argv == [
        "auth",
        "provision",
        "antrixj",
        "--output",
        "out.csv",
        "--password-length",
        "12",
    ]


def test_extract_env_files_errors_on_missing_value():
    with pytest.raises(SystemExit):
        extract_env_files(["auth", "--env-file"])


def test_parse_env_file_text_handles_comments_and_quotes():
    parsed = parse_env_file_text(
        """
        # comment
        FOO=bar
        BAR=baz # trailing comment
        QUOTED="a # not a comment"
        export EXPORTED=ok
        """
    )
    assert parsed["FOO"] == "bar"
    assert parsed["BAR"] == "baz"
    assert parsed["QUOTED"] == "a # not a comment"
    assert parsed["EXPORTED"] == "ok"


def test_load_env_files_overrides_in_order(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO", "old")
    one = tmp_path / "one.env"
    two = tmp_path / "two.env"
    one.write_text("FOO=first\nBAR=from_one\n", encoding="utf-8")
    two.write_text("FOO=second\n", encoding="utf-8")

    loaded = load_env_files([one, two], override=True)
    assert loaded["FOO"] == "second"
    assert loaded["BAR"] == "from_one"
    assert os.environ["FOO"] == "second"
    assert os.environ["BAR"] == "from_one"

