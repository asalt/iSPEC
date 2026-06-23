"""Link auth users to optional person records.

Auth users remain the login/access-control objects. Person records remain the
legacy/contact profile objects. This nullable link lets the UI connect them
without making either side mandatory.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0007_auth_user_person_link"
down_revision = "0006_experiment_run_legacy_modified"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def _index_names(table_name: str) -> set[str]:
    bind = op.get_bind()
    inspector = inspect(bind)
    return {idx["name"] for idx in inspector.get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("auth_user")}

    if "person_id" not in columns:
        with op.batch_alter_table("auth_user") as batch:
            batch.add_column(
                sa.Column(
                    "person_id",
                    sa.Integer(),
                    sa.ForeignKey("person.id"),
                    nullable=True,
                )
            )

    if "ix_auth_user_person_id" not in _index_names("auth_user"):
        op.create_index("ix_auth_user_person_id", "auth_user", ["person_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("auth_user")}

    if "ix_auth_user_person_id" in _index_names("auth_user"):
        op.drop_index("ix_auth_user_person_id", table_name="auth_user")

    if "person_id" in columns:
        with op.batch_alter_table("auth_user") as batch:
            batch.drop_column("person_id")
