"""Add label fields to experiment_run and extend uniqueness.

The legacy iSPEC ExperimentRuns table is effectively unique on:
  (experiment_id, run_no, search_no, label)

where ``label`` is a channel identifier ("0" for label-free; "127C", etc.).
``label_type`` captures the labeling strategy ("LabelFree", "TMT10", "SILAC", ...).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "0004_experiment_run_label"
down_revision = "0003_experiment_project_nullable"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    uniques = inspector.get_unique_constraints("experiment_run")
    desired_unique_cols = {"experiment_id", "run_no", "search_no", "label"}
    legacy_unique_cols = {"experiment_id", "run_no", "search_no"}

    has_desired_unique = any(
        set((uc.get("column_names") or [])) == desired_unique_cols for uc in uniques
    )
    if {"label", "label_type"}.issubset(columns) and has_desired_unique:
        return

    with op.batch_alter_table("experiment_run") as batch:
        if "label" not in columns:
            batch.add_column(
                sa.Column("label", sa.Text(), nullable=False, server_default=sa.text("'0'"))
            )
        if "label_type" not in columns:
            batch.add_column(sa.Column("label_type", sa.Text(), nullable=True))

        if not has_desired_unique:
            for uc in uniques:
                name = uc.get("name")
                cols = set((uc.get("column_names") or []))
                if name and cols == legacy_unique_cols:
                    batch.drop_constraint(name, type_="unique")
                    break

            batch.create_unique_constraint(
                "uq_experiment_run_search_label",
                ["experiment_id", "run_no", "search_no", "label"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("experiment_run")}
    uniques = inspector.get_unique_constraints("experiment_run")
    desired_unique_cols = {"experiment_id", "run_no", "search_no", "label"}
    legacy_unique_cols = {"experiment_id", "run_no", "search_no"}

    desired_unique_name = None
    has_legacy_unique = False
    for uc in uniques:
        cols = set((uc.get("column_names") or []))
        name = uc.get("name")
        if cols == legacy_unique_cols:
            has_legacy_unique = True
            legacy_unique_name = name
        if cols == desired_unique_cols:
            desired_unique_name = name

    with op.batch_alter_table("experiment_run") as batch:
        if desired_unique_name:
            batch.drop_constraint(desired_unique_name, type_="unique")

        if not has_legacy_unique:
            batch.create_unique_constraint(
                "uq_experiment_run_search",
                ["experiment_id", "run_no", "search_no"],
            )

        if "label_type" in columns:
            batch.drop_column("label_type")
        if "label" in columns:
            batch.drop_column("label")
