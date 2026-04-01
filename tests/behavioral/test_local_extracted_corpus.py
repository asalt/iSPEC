from __future__ import annotations

import pytest

from tests.behavioral.local_cases import assert_behavioral_case_expectations, load_local_behavioral_cases


pytestmark = pytest.mark.behavioral


def test_behavioral_local_extracted_corpus_validates_and_applies_expectations():
    cases = load_local_behavioral_cases()
    if not cases:
        pytest.skip('No local extracted behavioral cases found under tests/behavioral/local.')

    labels: set[str] = set()
    for case in cases:
        label = str(case.get('label') or '')
        assert label not in labels
        labels.add(label)
        assert case['messages']
        assert_behavioral_case_expectations(case)
