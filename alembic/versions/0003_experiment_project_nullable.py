"""Allow experiments without a project.

Legacy iSPEC data occasionally contains experiments that are not linked to a
project yet. The application should represent these as orphan experiments
instead of forcing a sentinel project id (e.g. 0).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0003_experiment_project_nullable"
down_revision = "0002_psm_msrawfile"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    with op.batch_alter_table("experiment") as batch:
        batch.alter_column(
            "project_id",
            existing_type=sa.Integer(),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("experiment") as batch:
        batch.alter_column(
            "project_id",
            existing_type=sa.Integer(),
            nullable=False,
        )

