"""Persist observed source values and unresolved ExperimentRun conflicts."""

from alembic import op
import sqlalchemy as sa

revision = "0009_run_source_baseline"
down_revision = "0008_experiment_run_legacy_status"
branch_labels = None
depends_on = None

_COLUMNS = ("legacy_source_baseline_json", "legacy_source_conflicts_json")


def upgrade() -> None:
    columns = {
        col["name"] for col in sa.inspect(op.get_bind()).get_columns("experiment_run")
    }
    with op.batch_alter_table("experiment_run") as batch:
        for name in _COLUMNS:
            if name not in columns:
                batch.add_column(sa.Column(name, sa.Text(), nullable=True))
    # Do not infer historical source values from existing local rows.


def downgrade() -> None:
    columns = {
        col["name"] for col in sa.inspect(op.get_bind()).get_columns("experiment_run")
    }
    with op.batch_alter_table("experiment_run") as batch:
        for name in _COLUMNS:
            if name in columns:
                batch.drop_column(name)
