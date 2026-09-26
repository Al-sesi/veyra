"""
Database persistence for Veyra.

Supports both PostgreSQL (via DATABASE_URL environment variable) and SQLite
(fallback for local development and testing). Uses psycopg2 for Postgres
and stdlib sqlite3 for SQLite.

Tables:
  - users   : id (PK), phone_or_name, language, created_at
  - entries : id (PK), user_id (FK users.id), item, quantity (nullable REAL),
              amount (INTEGER naira), type ("sale"/"expense"), transcript,
              audio_file, status ("active"/"voided"), replaces_entry_id,
              created_at
              Entries are never physically deleted: corrections/deletes flip
              `status` to "voided" so the audit trail survives.
  - debts   : id (PK), user_id (FK users.id), person, amount (INTEGER naira),
              direction ("owed_to_me"/"i_owe"), status ("open"/"paid"),
              created_at, paid_at (nullable)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

# Database type detection
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)


def _resolve_default_db_path() -> Path:
    env_path = os.environ.get("VEYRA_DB_PATH")
    if env_path:
        return Path(env_path).resolve()
    return Path("ledger.db").resolve()


DEFAULT_DB_PATH = _resolve_default_db_path()


def _connect_sqlite(db_path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with foreign keys + dict-like row access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _connect_postgres(database_url: str):
    """Open a PostgreSQL connection using psycopg2."""
    import psycopg2
    import psycopg2.extras
    
    conn = psycopg2.connect(database_url)
    # Return dict-like rows
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


@contextmanager
def get_connection(db_path: str | Path | None = None) -> Iterator:
    """Context manager yielding a live database connection (Postgres or SQLite).

    Commits on clean exit, rolls back if an exception is raised, and always
    closes the connection.

    `db_path=None` means DEFAULT_DB_PATH for SQLite, or DATABASE_URL for Postgres.
    If DATABASE_URL is set, Postgres is used regardless of db_path parameter.
    """
    if USE_POSTGRES:
        conn = _connect_postgres(DATABASE_URL)
    else:
        conn = _connect_sqlite(DEFAULT_DB_PATH if db_path is None else db_path)
    
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# SQLite schema (uses SQLite-specific syntax)
SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_or_name  TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'en',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item        TEXT    NOT NULL,
    quantity    REAL,
    amount      INTEGER NOT NULL,
    type        TEXT    NOT NULL CHECK (type IN ('sale', 'expense')),
    transcript  TEXT    NOT NULL,
    audio_file  TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active'
                               CHECK (status IN ('active', 'voided')),
    replaces_entry_id INTEGER REFERENCES entries(id),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS debts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    person      TEXT    NOT NULL,
    amount      INTEGER NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
    status      TEXT    NOT NULL DEFAULT 'open'
                               CHECK (status IN ('open', 'paid')),
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    paid_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_user_status
    ON debts(user_id, status);
"""

# PostgreSQL schema (uses Postgres-specific syntax)
SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS users (
    id             SERIAL PRIMARY KEY,
    phone_or_name  TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'en',
    created_at     TEXT    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS entries (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item        TEXT    NOT NULL,
    quantity    REAL,
    amount      INTEGER NOT NULL,
    type        TEXT    NOT NULL CHECK (type IN ('sale', 'expense')),
    transcript  TEXT    NOT NULL,
    audio_file  TEXT    NOT NULL DEFAULT '',
    status      TEXT    NOT NULL DEFAULT 'active'
                               CHECK (status IN ('active', 'voided')),
    replaces_entry_id INTEGER REFERENCES entries(id),
    created_at  TEXT    NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS debts (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    person      TEXT    NOT NULL,
    amount      INTEGER NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
    status      TEXT    NOT NULL DEFAULT 'open'
                               CHECK (status IN ('open', 'paid')),
    created_at  TEXT    NOT NULL DEFAULT NOW(),
    paid_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_user_status
    ON debts(user_id, status);
"""


def _add_column_if_missing_sqlite(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
) -> None:
    """Tiny migration helper for SQLite: ALTER TABLE ... ADD COLUMN only when needed."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table});")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")


def _add_column_if_missing_postgres(
    conn,
    table: str,
    column: str,
    ddl: str,
) -> None:
    """Tiny migration helper for PostgreSQL: ALTER TABLE ... ADD COLUMN only when needed."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{column}';
        """)
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")


def init_db(db_path: str | Path | None = None) -> None:
    """Create tables/indexes if they don't already exist. Safe to call repeatedly."""
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            schema_sql = SCHEMA_POSTGRES
            migration_fn = _add_column_if_missing_postgres
        else:
            schema_sql = SCHEMA_SQLITE
            migration_fn = _add_column_if_missing_sqlite
        
        if USE_POSTGRES:
            # For Postgres, execute each statement separately
            with conn.cursor() as cur:
                for statement in schema_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        cur.execute(statement)
        else:
            conn.executescript(schema_sql)
        
        # Migration for entries tables created before `replaces_entry_id` existed.
        migration_fn(
            conn, "entries", "replaces_entry_id", "INTEGER REFERENCES entries(id)"
        )


def get_or_create_user(
    db_path: str | Path | None,
    user_id: Optional[int],
    *,
    phone_or_name: str = "unknown",
    language: str = "en",
) -> int:
    """Resolve a user_id to an existing user (or create a brand-new one).

    - If `user_id` is passed and the row exists, return it unchanged.
    - If `user_id` is passed but doesn't exist, create a matching placeholder
      row (useful for tests that seed an id beforehand).
    - If `user_id` is None, create a new user with the provided
      `phone_or_name`/`language` and return the new id.
    """
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                if user_id is not None:
                    cur.execute(
                        "SELECT id FROM users WHERE id = %s;", (int(user_id),)
                    )
                    row = cur.fetchone()
                    if row is not None:
                        return int(row["id"])
                    cur.execute(
                        "INSERT INTO users(id, phone_or_name, language) VALUES (%s, %s, %s) RETURNING id;",
                        (int(user_id), phone_or_name, language),
                    )
                    return int(cur.fetchone()["id"])
                cur.execute(
                    "INSERT INTO users(phone_or_name, language) VALUES (%s, %s) RETURNING id;",
                    (phone_or_name, language),
                )
                return int(cur.fetchone()["id"])
        else:
            if user_id is not None:
                row = conn.execute(
                    "SELECT id FROM users WHERE id = ?;", (int(user_id),)
                ).fetchone()
                if row is not None:
                    return int(row["id"])
                cur = conn.execute(
                    "INSERT INTO users(id, phone_or_name, language) VALUES (?, ?, ?);",
                    (int(user_id), phone_or_name, language),
                )
                return int(cur.lastrowid)
            cur = conn.execute(
                "INSERT INTO users(phone_or_name, language) VALUES (?, ?);",
                (phone_or_name, language),
            )
            return int(cur.lastrowid)


def lookup_user_by_phone(
    db_path: str | Path | None,
    phone_number: str,
) -> Optional[dict[str, Any]]:
    """Find a user row by exact (case-insensitive) phone_or_name match.

    Returns the user dict (with ``id``, ``phone_or_name``, ``language``,
    ``created_at``) or ``None`` when no match exists.  The persistent
    ledger link ``/ledger/{phone_number}.xlsx`` uses this to scope the
    returned Excel file to exactly one trader's own entries.
    """
    if not phone_number:
        return None
    needle = phone_number.strip()
    
    if USE_POSTGRES:
        sql = "SELECT * FROM users WHERE LOWER(phone_or_name) = LOWER(%s) LIMIT 1;"
        params = (needle,)
    else:
        sql = "SELECT * FROM users WHERE LOWER(phone_or_name) = LOWER(?) LIMIT 1;"
        params = (needle,)
    
    row = query_one(
        DEFAULT_DB_PATH if db_path is None else db_path,
        sql,
        params,
    )
    return row


def row_to_dict(row) -> dict[str, Any]:
    """Convert a database row (sqlite3.Row or psycopg2 RealDictRow) into a plain dict."""
    if isinstance(row, dict):
        # Already a dict (PostgreSQL RealDictCursor)
        return dict(row)
    # SQLite Row
    return {k: row[k] for k in row.keys()}


def query_rows(
    db_path: str | Path,
    sql: str,
    params: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as plain dicts (small datasets only)."""
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Convert SQL ? placeholders to %s for PostgreSQL
                postgres_sql = sql.replace('?', '%s')
                cur.execute(postgres_sql, tuple(params))
                return [row_to_dict(row) for row in cur.fetchall()]
        else:
            return [row_to_dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def query_one(
    db_path: str | Path,
    sql: str,
    params: Iterable[Any] = (),
) -> Optional[dict[str, Any]]:
    """Run a SELECT that returns at most one row."""
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Convert SQL ? placeholders to %s for PostgreSQL
                postgres_sql = sql.replace('?', '%s')
                cur.execute(postgres_sql, tuple(params))
                row = cur.fetchone()
                return row_to_dict(row) if row is not None else None
        else:
            row = conn.execute(sql, tuple(params)).fetchone()
            return row_to_dict(row) if row is not None else None
