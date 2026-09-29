"""Small tracked defaults for legacy API fields required by core sync paths."""

from __future__ import annotations


EXPERIMENT_RUN_COMPOSITE_PK_FIELDS = (
    "exprun_EXPRecNo",
    "exprun_EXPRunNo",
    "exprun_EXPSearchNo",
)

EXPERIMENT_RUN_STATUS_FIELD_MAP = {
    "exprun_Grouper_StartFLAG": "legacy_grouper_started",
    "exprun_Grouper_EndFLAG": "legacy_grouper_completed",
    "exprun_Grouper_FailedFLAG": "legacy_grouper_failed",
    "exprun_Grouper_Version": "legacy_grouper_version",
    "exprun_Grouper_FilterStamp": "legacy_grouper_filter_stamp",
    "exprun_Grouper_RefDatabase": "legacy_grouper_ref_database",
    "exprun_InputFileName": "legacy_grouper_input_file",
    "exprun_PSMCount_matched": "legacy_grouper_matched_psm_count",
    "exprun_PSMCount_unmatched": "legacy_grouper_unmatched_psm_count",
    "exprun_Import_StartFLAG": "legacy_e2g_import_started",
    "exprun_Import_EndFLAG": "legacy_e2g_import_completed",
    "exprun_Import_FixFLAG": "legacy_e2g_import_fix_requested",
    "exprun_ImportTS": "legacy_e2g_imported_at",
    "exprun_ImportErrorLog": "legacy_e2g_import_error",
}
