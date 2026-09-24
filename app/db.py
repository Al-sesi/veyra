"""
Minimal SQLite persistence for sabi-books.

Uses only the stdlib `sqlite3` module (no third-party ORMs). The default DB
file is `ledger.db` in the current working directory, but any path can be
used (the unit tests pass in a temporary file instead so tests are isolated
and don't touch real data).

Tables:
  - users   : id (PK), phone_or_name, language, created_at
  - entries : id (PK), user_id (FK users.id), item, quantity (nullable REAL),
              amount (INTEGER naira), type ("sale"/"expense"), transcript,
              audio_file, status ("active"/"voided"), created_at
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


DEFAULT_DB_PATH = Path("ledger.db").resolve()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with foreign keys + dict-like row access."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a live sqlite3 Connection.

    Commits on clean exit, rolls back if an exception is raised, and always
    closes the connection.
    """
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA_SQL = """
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
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);
"""


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create tables/indexes if they don't already exist. Safe to call repeatedly."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def get_or_create_user(
    db_path: str | Path,
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


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row into a plain JSON-serializable dict."""
    return {k: row[k] for k in row.keys()}


def query_rows(
    db_path: str | Path,
    sql: str,
    params: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Run a SELECT and return rows as plain dicts (small datasets only)."""
    with get_connection(db_path) as conn:
        return [row_to_dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def query_one(
    db_path: str | Path,
    sql: str,
    params: Iterable[Any] = (),
) -> Optional[dict[str, Any]]:
    """Run a SELECT that returns at most one row."""
    with get_connection(db_path) as conn:
        row = conn.execute(sql, tuple(params)).fetchone()
        return row_to_dict(row) if row is not None else None
