"""Add label fields to experiment_run and extend uniqueness.

The legacy iSPEC ExperimentRuns table is effectively unique on:
  (experiment_id, run_no, search_no, label)

where ``label`` is a channel identifier ("0" for label-free; "127C", etc.).
``label_type`` captures the labeling strategy ("LabelFree", "TMT10", "SILAC", ...).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0004_experiment_run_label"
down_revision = "0003_experiment_project_nullable"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    with op.batch_alter_table("experiment_run") as batch:
        batch.add_column(
            sa.Column("label", sa.Text(), nullable=False, server_default=sa.text("'0'"))
        )
        batch.add_column(sa.Column("label_type", sa.Text(), nullable=True))

        try:
            batch.drop_constraint("uq_experiment_run_search", type_="unique")
        except Exception:
            pass

        batch.create_unique_constraint(
            "uq_experiment_run_search_label",
            ["experiment_id", "run_no", "search_no", "label"],
        )


def downgrade() -> None:
    with op.batch_alter_table("experiment_run") as batch:
        try:
            batch.drop_constraint("uq_experiment_run_search_label", type_="unique")
        except Exception:
            pass

        batch.create_unique_constraint(
            "uq_experiment_run_search",
            ["experiment_id", "run_no", "search_no"],
        )

        batch.drop_column("label_type")
        batch.drop_column("label")

