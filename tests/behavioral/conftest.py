from __future__ import annotations

import pytest

from tests.behavioral.datastore import BehavioralDatastore, create_behavioral_datastore


@pytest.fixture
def behavioral_datastore(tmp_path) -> BehavioralDatastore:
    return create_behavioral_datastore(tmp_path / "behavioral")
