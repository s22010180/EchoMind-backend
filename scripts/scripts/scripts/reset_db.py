"""
DEVELOPMENT ONLY: drop and recreate all tables.

The schema changed (new columns/tables), and SQLAlchemy's create_all() never alters
existing tables, so run this once against your dev database (SQLite or Neon).
Existing user accounts are deleted. For production data use Alembic migrations.

    python -m scripts.reset_db --yes
"""
import sys

from app import models  # noqa: F401
from app.config import settings
from app.database import Base, engine

if "--yes" not in sys.argv:
    print(f"This will DELETE ALL DATA in: {engine.url.render_as_string(hide_password=True)}")
    print("Re-run with --yes to confirm.")
    sys.exit(1)
if settings.is_production:
    print("Refusing to run with ENVIRONMENT=production.")
    sys.exit(1)

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print("Database recreated.")
