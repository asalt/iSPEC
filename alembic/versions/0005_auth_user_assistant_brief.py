"""Add assistant brief to auth_user.

This stores a short assistant-facing description for each user, which can be
injected into support-chat context and edited by staff.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0005_auth_user_assistant_brief"
down_revision = "0004_experiment_run_label"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("auth_user")}
    if "assistant_brief" in columns:
        return

    with op.batch_alter_table("auth_user") as batch:
        batch.add_column(sa.Column("assistant_brief", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("auth_user")}
    if "assistant_brief" not in columns:
        return

    with op.batch_alter_table("auth_user") as batch:
        batch.drop_column("assistant_brief")
