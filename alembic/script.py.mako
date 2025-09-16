"""${message}"""

revision = ${repr(revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

from alembic import op
import sqlalchemy as sa  # noqa: F401


def upgrade() -> None:
    """Apply the migration."""
    pass


def downgrade() -> None:
    """Revert the migration."""
    pass
