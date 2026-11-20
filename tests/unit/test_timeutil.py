from datetime import datetime

import pytest

from ispec.timeutil import compact_utc, format_utc, parse_utc, utcnow


def test_storage_and_filename_sort_in_utc():
    source = parse_utc("2026-11-18T23:10:01.345-06:00")
    assert format_utc(source) == "2026-11-19T05:10:01.345000Z"
    assert parse_utc(format_utc(source)) == source
    assert compact_utc(source) == "20261119T051001Z"
    assert format_utc(source) < format_utc(parse_utc("2026-11-19T05:10:02Z"))
    assert utcnow().utcoffset().total_seconds() == 0
    with pytest.raises(ValueError):
        format_utc(datetime(2026, 11, 18))
    with pytest.raises(ValueError):
        parse_utc("2026-11-18T23:10:00")
