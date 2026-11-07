from ispec.assistant.draft_fact_review import review_draft_facts


def test_invented_date_is_advisory():
    review = review_draft_facts(draft="Buffer replaced on 2026-11-06; QC is pending and under review.",
                                sources=("The extraction buffer was replaced and QC is pending.",))
    assert review.advisory_only
    assert [fact.value for fact in review.unsupported] == ["2026-11-06"]
    # The unsupported status 'under review' has no numeric/date evidence.


def test_missing_count_is_a_review_lead_not_a_false_write_block():
    review = review_draft_facts(draft="Labeling has not started.", sources=(
        "Labeling has started on four pooled samples.", "Change the status: labeling has not started.",
    ))
    assert not review.unsupported
    assert [fact.value for fact in review.omitted] == ["4"]


def test_numeric_and_word_counts_and_project_alias_agree():
    review = review_draft_facts(draft="MSPC000057: 8 samples arrived Monday; sample AB12 awaits QC.",
                                sources=("Project 57: eight samples arrived Monday; sample AB12 awaits QC.",))
    assert not review.unsupported and not review.omitted


def test_sample_and_run_ids_are_not_matched_as_counts():
    review = review_draft_facts(draft="Sample AB13, run 58209_1_2.",
                                sources=("Sample AB12, run 58209_1_1.",))
    assert {fact.value for fact in review.unsupported} == {"ab13", "58209_1_2"}


def test_printer_chatter_and_wrong_relations_are_known_blind_spots():
    source = "Seven samples arrived; three await QC. The printer is jammed."
    review = review_draft_facts(draft="Three samples arrived; seven await QC. The printer is jammed.", sources=(source,))
    assert not review.unsupported and not review.omitted


def test_retracted_values_and_nonliteral_paraphrases_need_human_review():
    review = review_draft_facts(draft="Eleven samples arrived on Friday.", sources=(
        "Ten samples arrived on 2026-11-06.", "Actually eleven, not ten.",
    ))
    assert [fact.value for fact in review.unsupported] == ["friday"]
    assert {fact.value for fact in review.omitted} == {"10", "2026-11-06"}
