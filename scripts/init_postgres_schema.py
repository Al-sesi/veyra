#!/usr/bin/env python3
"""
Initialize PostgreSQL schema for Veyra in Supabase.

This script creates the required tables (users, entries, debts) and indexes
in a PostgreSQL database. It's designed to be run against a live Supabase
instance using the DATABASE_URL environment variable.

Usage:
    export DATABASE_URL="postgresql://postgres:[PASSWORD]@[PROJECT-REF].pooler.supabase.com:6543/postgres"
    python scripts/init_postgres_schema.py
    
    Or pass it as a command line argument:
    python scripts/init_postgres_schema.py "postgresql://postgres:[PASSWORD]@[PROJECT-REF].pooler.supabase.com:6543/postgres"
"""

import os
import sys
from pathlib import Path


def get_schema_sql():
    """Return the PostgreSQL schema SQL as a string."""
    return """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id             SERIAL PRIMARY KEY,
    phone_or_name  TEXT    NOT NULL,
    language       TEXT    NOT NULL DEFAULT 'en',
    created_at     TEXT    NOT NULL DEFAULT NOW()
);

-- Entries table
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

-- Debts table
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

-- Indexes
CREATE INDEX IF NOT EXISTS idx_entries_user_created
    ON entries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_entries_user_status
    ON entries(user_id, status);

CREATE INDEX IF NOT EXISTS idx_debts_user_status
    ON debts(user_id, status);
"""


def main():
    # Check for dry-run flag first
    dry_run = "--dry-run" in sys.argv or os.environ.get("DRY_RUN")
    
    # Check for DATABASE_URL from environment, command line, or .env file
    database_url = os.environ.get("DATABASE_URL")
    
    # Check command line argument (skip flags)
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            database_url = arg
            break
    
    # Check .env file
    if not database_url:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DATABASE_URL="):
                        database_url = line.split("=", 1)[1].strip()
                        break
    
    if not database_url:
        print("Error: DATABASE_URL is not set.")
        print("Please set it via:")
        print("  1. Environment variable: export DATABASE_URL=\"...\"")
        print("  2. Command line argument: python scripts/init_postgres_schema.py \"...\"")
        print("  3. .env file: DATABASE_URL=...")
        sys.exit(1)

    print(f"Connecting to PostgreSQL database...")
    # For security, only show protocol and host, not credentials
    try:
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        safe_display = f"{parsed.scheme}://[hidden]@{parsed.hostname}:{parsed.port}{parsed.path}"
        print(f"Using DATABASE_URL: {safe_display}")
    except:
        print(f"Using DATABASE_URL: {database_url[:30]}...{database_url[-10:]}")  # Fallback
    
    # Add a dry-run mode for testing without actual connection
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print("Simulating schema creation without actual database connection...")
        print("The following SQL would be executed:")
        print(get_schema_sql())
        print("\n=== END DRY RUN ===")
        print("Remove --dry-run to execute against real database.")
        return
    
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("Error: psycopg2 is not installed.")
        print("Install it with: pip install psycopg2-binary")
        sys.exit(1)

    try:
        # Add connection timeout and SSL mode for better debugging
        conn = psycopg2.connect(
            database_url,
            connect_timeout=10,
            sslmode='require'
        )
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        print("Connected successfully!")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the DATABASE_URL is correct")
        print("2. Check your internet connection")
        print("3. Verify the Supabase project is active")
        print("4. Try pinging the host to check connectivity")
        sys.exit(1)

    schema_sql = get_schema_sql()

    try:
        print("Creating tables and indexes...")
        # Execute each statement separately
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                cur.execute(statement)
        
        print("Schema created successfully!")
        
        # Verify tables exist
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row['table_name'] for row in cur.fetchall()]
        print(f"Tables in database: {', '.join(tables)}")
        
        # Verify indexes exist
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY indexname;
        """)
        indexes = [row['indexname'] for row in cur.fetchall()]
        print(f"Indexes created: {', '.join(indexes)}")
        
        cur.close()
        conn.close()
        print("\nDatabase initialization complete!")
        
    except Exception as e:
        print(f"Error creating schema: {e}")
        try:
            cur.close()
            conn.close()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
