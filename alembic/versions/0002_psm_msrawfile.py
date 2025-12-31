"""Add PSM and MSRawFile tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_psm_msrawfile"
down_revision = "0001_initial"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    op.create_table(
        "psm",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_run_id", sa.Integer(), nullable=False),
        sa.Column("scan_number", sa.Integer(), nullable=False),
        sa.Column("peptide", sa.Text(), nullable=False),
        sa.Column("charge", sa.Integer(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("score_type", sa.Text(), nullable=True),
        sa.Column("q_value", sa.Float(), nullable=True),
        sa.Column("protein", sa.Text(), nullable=True),
        sa.Column("mods", sa.Text(), nullable=True),
        sa.Column("precursor_mz", sa.Float(), nullable=True),
        sa.Column("retention_time", sa.Float(), nullable=True),
        sa.Column("intensity", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("psm_CreationTS", sa.DateTime(), nullable=False),
        sa.Column("psm_ModificationTS", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_run_id"], ["experiment_run.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "ms_raw_file",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_run_id", sa.Integer(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("checksum", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column(
            "file_type",
            sa.Enum(
                "raw",
                "mzml",
                "parquet",
                "bruker_d",
                "other",
                name="rawfiletype",
                native_enum=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "storage_backend",
            sa.Enum("local", "s3", "gcs", "other", name="storagebackend", native_enum=False),
            nullable=False,
        ),
        sa.Column(
            "state",
            sa.Enum("available", "archived", "missing", name="rawfilestate", native_enum=False),
            nullable=False,
        ),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("msraw_CreationTS", sa.DateTime(), nullable=False),
        sa.Column("msraw_ModificationTS", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_run_id"], ["experiment_run.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_run_id", "uri", name="uq_msraw_run_uri"),
    )


def downgrade() -> None:
    op.drop_table("ms_raw_file")
    op.drop_table("psm")
