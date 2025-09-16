"""Create initial iSPEC tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa  # noqa: F401

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Apply the initial schema using the SQLAlchemy models."""

    from ispec.db.models import Base

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop the schema created by :func:`upgrade`."""

    from ispec.db.models import Base

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
