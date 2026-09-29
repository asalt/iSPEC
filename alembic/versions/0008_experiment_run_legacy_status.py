"""Add legacy ExperimentRun processing metadata and composite cursor storage."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0008_experiment_run_legacy_status"
down_revision = "0007_auth_user_person_link"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


_EXPERIMENT_RUN_COLUMNS = (
    sa.Column("legacy_grouper_started", sa.Boolean(), nullable=True),
    sa.Column("legacy_grouper_completed", sa.Boolean(), nullable=True),
    sa.Column("legacy_grouper_failed", sa.Boolean(), nullable=True),
    sa.Column("legacy_grouper_version", sa.Text(), nullable=True),
    sa.Column("legacy_grouper_filter_stamp", sa.Text(), nullable=True),
    sa.Column("legacy_grouper_ref_database", sa.Text(), nullable=True),
    sa.Column("legacy_grouper_input_file", sa.Text(), nullable=True),
    sa.Column("legacy_grouper_matched_psm_count", sa.Integer(), nullable=True),
    sa.Column("legacy_grouper_unmatched_psm_count", sa.Integer(), nullable=True),
    sa.Column("legacy_e2g_import_started", sa.Boolean(), nullable=True),
    sa.Column("legacy_e2g_import_completed", sa.Boolean(), nullable=True),
    sa.Column("legacy_e2g_import_fix_requested", sa.Boolean(), nullable=True),
    sa.Column("legacy_e2g_imported_at", sa.DateTime(), nullable=True),
    sa.Column("legacy_e2g_import_error", sa.Text(), nullable=True),
)


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    run_columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    missing_run_columns = [
        col for col in _EXPERIMENT_RUN_COLUMNS if col.name not in run_columns
    ]
    if missing_run_columns:
        with op.batch_alter_table("experiment_run") as batch:
            for column in missing_run_columns:
                batch.add_column(column)

    sync_columns = {col["name"] for col in inspector.get_columns("legacy_sync_state")}
    if "since_pk_json" not in sync_columns:
        with op.batch_alter_table("legacy_sync_state") as batch:
            batch.add_column(sa.Column("since_pk_json", sa.Text(), nullable=True))


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    sync_columns = {col["name"] for col in inspector.get_columns("legacy_sync_state")}
    if "since_pk_json" in sync_columns:
        with op.batch_alter_table("legacy_sync_state") as batch:
            batch.drop_column("since_pk_json")

    run_columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    present_run_columns = [
        col.name for col in _EXPERIMENT_RUN_COLUMNS if col.name in run_columns
    ]
    if present_run_columns:
        with op.batch_alter_table("experiment_run") as batch:
            for column_name in present_run_columns:
                batch.drop_column(column_name)
