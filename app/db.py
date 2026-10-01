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
  - llm_extraction_logs : id (PK), user_id (FK users.id), transcript, language,
                         raw_llm_output, entries_extracted, confidence,
                         rule_based_parser_tried_first, rule_based_parser_failed,
                         timestamp
                         Logs LLM extraction attempts for analysis and documentation.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

# Load environment variables from .env file (skip if TEST_MODE is set)
if not os.environ.get("TEST_MODE"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv is optional

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
    detected_language TEXT NOT NULL DEFAULT 'en',
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

CREATE TABLE IF NOT EXISTS debtors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    canonical_name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, canonical_name)
);

CREATE TABLE IF NOT EXISTS debt_transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    debtor_id   INTEGER NOT NULL REFERENCES debtors(id) ON DELETE CASCADE,
    amount      INTEGER NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('debt', 'payment')),
    notes       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_extraction_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER,
    transcript  TEXT    NOT NULL,
    language    TEXT    NOT NULL,
    raw_llm_output TEXT,
    entries_extracted TEXT,
    confidence TEXT,
    rule_based_parser_tried_first INTEGER NOT NULL DEFAULT 1,
    rule_based_parser_failed INTEGER NOT NULL DEFAULT 0,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_user_status
    ON debts(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debtors_user_canonical
    ON debtors(user_id, canonical_name);

CREATE INDEX IF NOT EXISTS idx_debt_transactions_debtor
    ON debt_transactions(debtor_id);

CREATE INDEX IF NOT EXISTS idx_llm_logs_user_timestamp
    ON llm_extraction_logs(user_id, timestamp DESC);
"""

# PostgreSQL schema (uses Postgres-specific syntax)
SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS users (
    id             SERIAL PRIMARY KEY,
    phone_or_name  TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'en',
    created_at     TIMESTAMP NOT NULL DEFAULT NOW()
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
    detected_language TEXT NOT NULL DEFAULT 'en',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS debts (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    person      TEXT    NOT NULL,
    amount      INTEGER NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
    status      TEXT    NOT NULL DEFAULT 'open'
                               CHECK (status IN ('open', 'paid')),
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    paid_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS debtors (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    canonical_name TEXT NOT NULL,
    original_name TEXT NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, canonical_name)
);

CREATE TABLE IF NOT EXISTS debt_transactions (
    id          SERIAL PRIMARY KEY,
    debtor_id   INTEGER NOT NULL REFERENCES debtors(id) ON DELETE CASCADE,
    amount      INTEGER NOT NULL,
    direction   TEXT    NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('debt', 'payment')),
    notes       TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS llm_extraction_logs (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER,
    transcript  TEXT    NOT NULL,
    language    TEXT    NOT NULL,
    raw_llm_output TEXT,
    entries_extracted TEXT,
    confidence TEXT,
    rule_based_parser_tried_first INTEGER NOT NULL DEFAULT 1,
    rule_based_parser_failed INTEGER NOT NULL DEFAULT 0,
    timestamp   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_user_status
    ON debts(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debtors_user_canonical
    ON debtors(user_id, canonical_name);

CREATE INDEX IF NOT EXISTS idx_debt_transactions_debtor
    ON debt_transactions(debtor_id);

CREATE INDEX IF NOT EXISTS idx_llm_logs_user_timestamp
    ON llm_extraction_logs(user_id, timestamp DESC);
"""


def _add_column_if_missing_sqlite(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
    *,
    full_ddl: str = "",
    is_index: bool = False,
) -> None:
    """Tiny migration helper for SQLite: ALTER TABLE ... ADD COLUMN only when needed.
    
    If full_ddl is provided, creates the entire table if it doesn't exist.
    If is_index is True, creates an index if it doesn't exist.
    """
    if full_ddl:
        # Create table if it doesn't exist
        existing_tables = {
            row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        }
        if table not in existing_tables:
            conn.executescript(full_ddl)
        return
    
    if is_index:
        # Create index if it doesn't exist
        existing_indexes = {
            row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index';")
        }
        if column not in existing_indexes:
            conn.execute(f"CREATE INDEX IF NOT EXISTS {column} ON {ddl};")
        return
    
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table});")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")


def _add_column_if_missing_postgres(
    conn,
    table: str,
    column: str,
    ddl: str,
    *,
    full_ddl: str = "",
    is_index: bool = False,
) -> None:
    """Tiny migration helper for PostgreSQL: ALTER TABLE ... ADD COLUMN only when needed.
    
    If full_ddl is provided, creates the entire table if it doesn't exist.
    If is_index is True, creates an index if it doesn't exist.
    """
    if full_ddl:
        # Create table if it doesn't exist
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = %s;
            """, (table,))
            if cur.fetchone() is None:
                cur.execute(full_ddl)
        return
    
    if is_index:
        # Create index if it doesn't exist
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT indexname 
                FROM pg_indexes 
                WHERE indexname = '{column}';
            """)
            if cur.fetchone() is None:
                cur.execute(f"CREATE INDEX IF NOT EXISTS {column} ON {ddl};")
        return
    
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{column}';
        """)
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl};")


def init_db(db_path: str | Path | None = None) -> None:
    """Create tables/indexes if they don't already exist. Safe to call repeatedly.
    
    For PostgreSQL (DATABASE_URL set), db_path is ignored and DATABASE_URL is used.
    For SQLite (DATABASE_URL not set), db_path defaults to DEFAULT_DB_PATH if not provided.
    """
    # For PostgreSQL, ignore db_path parameter and use DATABASE_URL
    if USE_POSTGRES:
        db_path = None
    elif db_path is None:
        db_path = DEFAULT_DB_PATH
        
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
        
        # Migration for entries tables created before `detected_language` existed.
        migration_fn(
            conn, "entries", "detected_language", "TEXT NOT NULL DEFAULT 'en'"
        )
        
        # Migration for debtors and debt_transactions tables (new debt tracking system)
        # These tables don't exist in older databases, so we create them if missing
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Create debtors table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS debtors (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        canonical_name TEXT NOT NULL,
                        original_name TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                        UNIQUE(user_id, canonical_name)
                    );
                """)
                # Create debt_transactions table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS debt_transactions (
                        id SERIAL PRIMARY KEY,
                        debtor_id INTEGER NOT NULL REFERENCES debtors(id) ON DELETE CASCADE,
                        amount INTEGER NOT NULL,
                        direction TEXT NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
                        transaction_type TEXT NOT NULL CHECK (transaction_type IN ('debt', 'payment')),
                        notes TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                """)
                # Create indexes if they don't exist
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_debtors_user_canonical
                    ON debtors(user_id, canonical_name);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_debt_transactions_debtor
                    ON debt_transactions(debtor_id);
                """)
        else:
            # SQLite migration
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS debtors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    canonical_name TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(user_id, canonical_name)
                );
                
                CREATE TABLE IF NOT EXISTS debt_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    debtor_id INTEGER NOT NULL REFERENCES debtors(id) ON DELETE CASCADE,
                    amount INTEGER NOT NULL,
                    direction TEXT NOT NULL CHECK (direction IN ('owed_to_me', 'i_owe')),
                    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('debt', 'payment')),
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                
                CREATE INDEX IF NOT EXISTS idx_debtors_user_canonical
                ON debtors(user_id, canonical_name);
                
                CREATE INDEX IF NOT EXISTS idx_debt_transactions_debtor
                ON debt_transactions(debtor_id);
            """)


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


def log_llm_extraction(
    user_id: Optional[int],
    transcript: str,
    language: str,
    raw_llm_output: str,
    entries_extracted: list[dict[str, Any]],
    confidence: str,
    rule_based_parser_tried_first: bool,
    rule_based_parser_failed: bool,
    *,
    db_path: str | Path | None = None,
) -> None:
    """Log an LLM extraction attempt to the llm_extraction_logs table.
    
    This function is used to track LLM fallback extractions for analysis
    and documentation purposes. It does not log API keys or other secrets.
    
    Parameters
    ----------
    user_id : Optional[int]
        The user ID (can be None if user not yet created)
    transcript : str
        The original transcript
    language : str
        Language code
    raw_llm_output : str
        Raw response from the LLM (JSON string)
    entries_extracted : list[dict]
        The parsed entries from the LLM
    confidence : str
        Overall confidence level ("high" or "low")
    rule_based_parser_tried_first : bool
        Whether the rule-based parser was tried first
    rule_based_parser_failed : bool
        Whether the rule-based parser failed
    db_path : str | Path | None
        Database path (defaults to DEFAULT_DB_PATH)
    """
    import json
    
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Convert entries to JSON string for storage
    entries_json = json.dumps(entries_extracted)
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_extraction_logs 
                    (user_id, transcript, language, raw_llm_output, entries_extracted, 
                     confidence, rule_based_parser_tried_first, rule_based_parser_failed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (user_id, transcript, language, raw_llm_output, entries_json,
                     confidence, rule_based_parser_tried_first, rule_based_parser_failed)
                )
        else:
            conn.execute(
                """
                INSERT INTO llm_extraction_logs 
                (user_id, transcript, language, raw_llm_output, entries_extracted, 
                 confidence, rule_based_parser_tried_first, rule_based_parser_failed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (user_id, transcript, language, raw_llm_output, entries_json,
                 confidence, rule_based_parser_tried_first, rule_based_parser_failed)
            )
