"""Create the local MySQL database, then create the three SQLAlchemy tables."""
from sqlalchemy import inspect, text

from database import database_settings, get_engine, make_engine
from models import Base


def initialize():
    name = database_settings()["database"]
    server = make_engine(include_database=False)
    try:
        with server.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            # The identifier is strictly validated in database_settings().
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    finally:
        server.dispose()
    engine = get_engine()
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = set(Base.metadata.tables)
    if not expected <= set(tables):
        raise RuntimeError("Expected tables are missing after initialization")
    print(f"Database ready: {name}")
    for table in sorted(expected):
        print(f"Table: {table} ({len(inspector.get_columns(table))} columns)")
    print("Existing data was preserved. Timestamps are stored in UTC.")


if __name__ == "__main__":
    try:
        initialize()
    except Exception as exc:
        # Never print connection URLs or driver errors that could expose credentials.
        print(f"Database initialization failed ({type(exc).__name__}). Check MySQL service and MYSQL_* settings.")
        raise SystemExit(1) from None
