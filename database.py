"""SQLAlchemy configuration for local MySQL chat storage."""
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import sessionmaker


def database_settings():
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    name = os.getenv("MYSQL_DATABASE", "rag_chat")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
        raise ValueError("MYSQL_DATABASE must be a simple alphanumeric/underscore name")
    password = os.getenv("MYSQL_PASSWORD")
    if password is None:
        raise ValueError("Set MYSQL_PASSWORD in .env")
    return {"host": os.getenv("MYSQL_HOST", "localhost"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "username": os.getenv("MYSQL_USER", "root"),
            "password": password, "database": name}


def make_engine(include_database=True):
    settings = database_settings()
    if not include_database:
        settings.pop("database")
    # URL.create handles special password characters without string interpolation.
    url = URL.create("mysql+mysqlconnector", **settings)
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800, echo=False,
                         hide_parameters=True,
                         connect_args={"use_pure": True, "charset": "utf8mb4", "connection_timeout": 10})


@lru_cache(maxsize=1)
def get_engine():
    return make_engine()


@lru_cache(maxsize=1)
def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db():
    """FastAPI dependency: close sessions automatically; callers commit writes."""
    with get_session_factory()() as session:
        yield session
