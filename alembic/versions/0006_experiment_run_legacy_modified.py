"""Add legacy modified timestamp to experiment_run.

The existing ExperimentRun_ModificationTS column is app-managed locally. This
column preserves the source legacy row modification timestamp from FileMaker.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0006_experiment_run_legacy_modified"
down_revision = "0005_auth_user_assistant_brief"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    if "ExperimentRun_LegacyModificationTS" in columns:
        return

    with op.batch_alter_table("experiment_run") as batch:
        batch.add_column(
            sa.Column("ExperimentRun_LegacyModificationTS", sa.DateTime(), nullable=True)
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    if "ExperimentRun_LegacyModificationTS" not in columns:
        return

    with op.batch_alter_table("experiment_run") as batch:
        batch.drop_column("ExperimentRun_LegacyModificationTS")
