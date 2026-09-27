#!/usr/bin/env python3
"""
Migration script to fix TEXT created_at columns to TIMESTAMP in PostgreSQL.

This script migrates existing Veyra PostgreSQL databases from TEXT to TIMESTAMP
for created_at and paid_at columns, which is required for proper date comparisons.
"""

import os
import sys
from pathlib import Path


def main():
    # Check for force flag
    force = "--force" in sys.argv or os.environ.get("FORCE_MIGRATION")
    
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
        print("  2. Command line argument: python scripts/fix_postgres_timestamps.py \"...\"")
        print("  3. .env file: DATABASE_URL=...")
        sys.exit(1)

    print(f"Connecting to PostgreSQL database...")
    try:
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        safe_display = f"{parsed.scheme}://[hidden]@{parsed.hostname}:{parsed.port}{parsed.path}"
        print(f"Using DATABASE_URL: {safe_display}")
    except:
        print(f"Using DATABASE_URL: {database_url[:30]}...{database_url[-10:]}")
    
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("Error: psycopg2 is not installed.")
        print("Install it with: pip install psycopg2-binary")
        sys.exit(1)

    try:
        conn = psycopg2.connect(
            database_url,
            connect_timeout=10,
            sslmode='require'
        )
        conn.autocommit = False  # Use transactions for safety
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        print("Connected successfully!")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    try:
        print("\nChecking current schema...")
        
        # Check if columns are TEXT
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name IN ('users', 'entries', 'debts')
              AND column_name IN ('created_at', 'paid_at')
            ORDER BY table_name, column_name;
        """)
        columns = cur.fetchall()
        
        text_columns = [row for row in columns if row['data_type'] == 'text']
        
        if not text_columns:
            print("All timestamp columns are already using proper data types.")
            print("Migration not needed.")
            cur.close()
            conn.close()
            return
        
        print(f"Found {len(text_columns)} TEXT columns that need migration:")
        for col in text_columns:
            print(f"  - {col['column_name']} in table (inferred from context)")
        
        if not force:
            print("\nWARNING: This will modify existing table columns.")
            print("Use --force flag to proceed without confirmation.")
            response = input("Do you want to proceed? (yes/no): ")
            if response.lower() != 'yes':
                print("Migration cancelled.")
                cur.close()
                conn.close()
                return
        else:
            print("\nForce mode enabled - proceeding with migration...")
        
        print("\nStarting migration...")
        
        # For each table, alter the columns
        migrations = [
            ("users", "created_at"),
            ("entries", "created_at"),
            ("debts", "created_at"),
            ("debts", "paid_at"),
        ]
        
        for table, column in migrations:
            try:
                # First, create a temporary column with the correct type
                temp_col = f"{column}_temp"
                cur.execute(f"""
                    ALTER TABLE {table} 
                    ADD COLUMN IF NOT EXISTS {temp_col} TIMESTAMP;
                """)
                
                # Copy data from old column to new column (convert TEXT to TIMESTAMP)
                cur.execute(f"""
                    UPDATE {table} 
                    SET {temp_col} = {column}::TIMESTAMP 
                    WHERE {column} IS NOT NULL;
                """)
                
                # Drop the old column
                cur.execute(f"""
                    ALTER TABLE {table} 
                    DROP COLUMN IF EXISTS {column};
                """)
                
                # Rename the temporary column to the original name
                cur.execute(f"""
                    ALTER TABLE {table} 
                    RENAME COLUMN {temp_col} TO {column};
                """)
                
                print(f"Migrated {table}.{column} to TIMESTAMP")
                
            except Exception as e:
                print(f"Error migrating {table}.{column}: {e}")
                conn.rollback()
                cur.close()
                conn.close()
                sys.exit(1)
        
        conn.commit()
        print("\nMigration completed successfully!")
        
        # Verify the changes
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name IN ('users', 'entries', 'debts')
              AND column_name IN ('created_at', 'paid_at')
            ORDER BY table_name, column_name;
        """)
        columns = cur.fetchall()
        
        print("\nUpdated schema:")
        for col in columns:
            print(f"  {col['column_name']}: {col['data_type']}")
        
        cur.close()
        conn.close()
        print("\nMigration complete!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
        try:
            cur.close()
            conn.close()
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
