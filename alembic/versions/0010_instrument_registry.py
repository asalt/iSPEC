"""Staff instrument registry; archives remain independent databases."""

from alembic import op
import sqlalchemy as sa

revision = "0010_instrument_registry"
down_revision = "0009_run_source_baseline"
branch_labels = depends_on = None


def upgrade():
    if "instrument_registry" not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(
            "instrument_registry",
            sa.Column("id", sa.String(), primary_key=True),
            sa.Column("settings", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("updated_at", sa.String(), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("archive_seen", sa.Boolean(), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
        )


def downgrade():
    op.drop_table("instrument_registry")
