"""
Ledger operations backed by app.db.

Entry operations:
  - add_entries(user_id, entries, transcript, audio_file)
  - list_entries(user_id, limit=50)
  - get_summary(user_id, days=7)
  - correct_last_entry(user_id, new_amount)
  - delete_last_entry(user_id)

Debt operations:
  - add_debt(user_id, person, amount, direction)
  - mark_debt_paid(user_id, person)
  - list_open_debts(user_id)

`get_summary` returns total sales, total expenses, profit, the top sale
item (by summed amount), the top expense item (by summed amount), and the
open-debt totals ("total_owed_to_me" / "total_i_owe"). Only "active"-status
entries contribute; voided rows are kept for the audit trail but ignored.

Corrections and deletes NEVER physically delete a row: the old row is
flipped to status "voided", and a correction inserts a fresh "active" row
that points back at it via `replaces_entry_id`.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Optional

from io import BytesIO
from urllib.parse import quote

from app.db import DEFAULT_DB_PATH, USE_POSTGRES, get_connection, get_or_create_user, query_one, row_to_dict


DEBT_DIRECTIONS = ("owed_to_me", "i_owe")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_user(db_path, user_id):
    """Ensure a user row exists; return the int id (create a placeholder if needed)."""
    return get_or_create_user(
        db_path,
        user_id,
        phone_or_name=f"user_{user_id}" if user_id is not None else "unknown",
        language="en",
    )


def _last_active_entry(
    conn,
    user_id: int,
) -> Optional:
    """The most recent active entry for a user, or None."""
    if USE_POSTGRES:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM entries
                WHERE user_id = %s AND status = 'active'
                ORDER BY created_at DESC, id DESC
                LIMIT 1;
                """,
                (user_id,),
            )
            return cur.fetchone()
    else:
        return conn.execute(
            """
            SELECT * FROM entries
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC, id DESC
            LIMIT 1;
            """,
            (user_id,),
        ).fetchone()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_entries(
    user_id: Optional[int],
    entries: Iterable[dict[str, Any]],
    transcript: str,
    audio_file: str = "",
    *,
    db_path=None,
    detected_language: str = "en",
) -> list[dict[str, Any]]:
    """Insert a batch of parsed entries for a user, attach transcript+audio_file metadata
    on each row. Returns the freshly inserted rows (with generated ids)."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    entries = list(entries)
    if not entries:
        return []
    uid = _require_user(db_path, user_id)

    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                insert_sql = """
                    INSERT INTO entries (user_id, item, quantity, amount, type,
                                     transcript, audio_file, status, detected_language)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """
                inserted_ids = []
                for e in entries:
                    qty = e.get("quantity")
                    cur.execute(
                        insert_sql,
                        (
                            uid,
                            str(e.get("item", "") or "").strip() or "unknown",
                            None if qty is None else float(qty),
                            int(e["amount"]),
                            str(e.get("type", "sale")),
                            transcript or "",
                            audio_file or "",
                            "active",
                            detected_language,
                        )
                    )
                    inserted_ids.append(cur.fetchone()["id"])
                
                # Fetch the inserted rows
                if inserted_ids:
                    placeholders = ','.join(['%s'] * len(inserted_ids))
                    cur.execute(
                        f"""
                        SELECT * FROM entries
                        WHERE id IN ({placeholders}) AND user_id = %s
                        ORDER BY id ASC;
                        """,
                        inserted_ids + [uid]
                    )
                    return [row_to_dict(row) for row in cur.fetchall()]
                return []
        else:
            insert_sql = """
                INSERT INTO entries (user_id, item, quantity, amount, type,
                                 transcript, audio_file, status, detected_language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            rows: list[tuple] = []
            for e in entries:
                qty = e.get("quantity")
                rows.append(
                    (
                        uid,
                        str(e.get("item", "") or "").strip() or "unknown",
                        None if qty is None else float(qty),
                        int(e["amount"]),
                        str(e.get("type", "sale")),
                        transcript or "",
                        audio_file or "",
                        "active",
                        detected_language,
                    )
                )
            conn.executemany(insert_sql, rows)
            # Pull back the inserted rows in insertion order. SQLite's
            # last_insert_rowid() reports the id of the LAST row inserted by
            # executemany; the batch occupies the contiguous ids before it.
            last_id = conn.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"]
            first_id = last_id - len(rows) + 1
            fetched = conn.execute(
                """
                SELECT * FROM entries
                WHERE id BETWEEN ? AND ? AND user_id = ?
                ORDER BY id ASC;
                """,
                (first_id, last_id, uid),
            ).fetchall()
            return [row_to_dict(r) for r in fetched]


def list_entries(
    user_id: int,
    limit: int = 50,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """Return the most recent `limit` entries for a user (all statuses)."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM entries
                    WHERE user_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s;
                    """,
                    (uid, int(limit)),
                )
                return [row_to_dict(row) for row in cur.fetchall()]
        else:
            rows = conn.execute(
                """
                SELECT * FROM entries
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?;
                """,
                (uid, int(limit)),
            ).fetchall()
            return [row_to_dict(r) for r in rows]


def correct_last_entry(
    user_id: int,
    new_amount: int,
    *,
    db_path=None,
) -> Optional[dict[str, Any]]:
    """Fix the amount of the user's most recent active entry.

    The old row is voided (kept as an audit trail) and replaced by a fresh
    active row copying item/quantity/type/transcript/audio_file with the new
    amount, linked back via `replaces_entry_id`.

    Returns the new active row, or None when there is nothing to correct.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
        old = _last_active_entry(conn, uid)
        if old is None:
            return None
        
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE entries SET status = 'voided' WHERE id = %s;", (old["id"],)
                )
                cur.execute(
                    """
                    INSERT INTO entries (user_id, item, quantity, amount, type,
                                         transcript, audio_file, status,
                                         replaces_entry_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'active', %s)
                    RETURNING id;
                    """,
                    (
                        uid,
                        old["item"],
                        old["quantity"],
                        int(new_amount),
                        old["type"],
                        old["transcript"],
                        old["audio_file"],
                        old["id"],
                    ),
                )
                new_id = cur.fetchone()["id"]
                cur.execute(
                    "SELECT * FROM entries WHERE id = %s;", (new_id,)
                )
                return row_to_dict(cur.fetchone())
        else:
            conn.execute(
                "UPDATE entries SET status = 'voided' WHERE id = ?;", (old["id"],)
            )
            cur = conn.execute(
                """
                INSERT INTO entries (user_id, item, quantity, amount, type,
                                     transcript, audio_file, status,
                                     replaces_entry_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?);
                """,
                (
                    uid,
                    old["item"],
                    old["quantity"],
                    int(new_amount),
                    old["type"],
                    old["transcript"],
                    old["audio_file"],
                    old["id"],
                ),
            )
            new_row = conn.execute(
                "SELECT * FROM entries WHERE id = ?;", (cur.lastrowid,)
            ).fetchone()
            return row_to_dict(new_row)


def delete_last_entry(
    user_id: int,
    *,
    db_path=None,
) -> Optional[dict[str, Any]]:
    """Void (not delete) the user's most recent active entry.

    Returns the now-voided row, or None when there is nothing to remove.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
        last = _last_active_entry(conn, uid)
        if last is None:
            return None
        
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE entries SET status = 'voided' WHERE id = %s;", (last["id"],)
                )
                cur.execute(
                    "SELECT * FROM entries WHERE id = %s;", (last["id"],)
                )
                return row_to_dict(cur.fetchone())
        else:
            conn.execute(
                "UPDATE entries SET status = 'voided' WHERE id = ?;", (last["id"],)
            )
            voided = conn.execute(
                "SELECT * FROM entries WHERE id = ?;", (last["id"],)
            ).fetchone()
            return row_to_dict(voided)


# ---------------------------------------------------------------------------
# Debts
# ---------------------------------------------------------------------------

def add_debt(
    user_id: Optional[int],
    person: str,
    amount: int,
    direction: str,
    *,
    db_path=None,
) -> dict[str, Any]:
    """Record a debt using the new debtor identity system.
    
    `direction` is "owed_to_me" (they owe the trader) or
    "i_owe" (the trader owes them). 
    
    This function:
    1. Finds or creates a debtor record by exact name match
    2. Adds a debt transaction to that debtor
    3. Also records in the legacy debts table for backward compatibility
    
    Returns a dict with person, amount, direction, and other fields for backward compatibility.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if direction not in DEBT_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DEBT_DIRECTIONS}, got {direction!r}"
        )
    uid = _require_user(db_path, user_id)
    name = (person or "").strip() or "Unknown"
    
    # Get or create debtor using exact name match
    debtor = _get_or_create_debtor(uid, name, db_path=db_path)
    
    # Add debt transaction
    transaction = add_debt_transaction(
        uid, debtor["id"], amount, direction, transaction_type="debt", db_path=db_path
    )
    
    # Also record in legacy debts table for backward compatibility
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO debts (user_id, person, amount, direction, status)
                    VALUES (%s, %s, %s, %s, 'open')
                    RETURNING id;
                    """,
                    (uid, name, int(amount), direction),
                )
                legacy_id = cur.fetchone()["id"]
        else:
            cur = conn.execute(
                """
                INSERT INTO debts (user_id, person, amount, direction, status)
                VALUES (?, ?, ?, ?, 'open');
                """,
                (uid, name, int(amount), direction),
            )
            legacy_id = cur.lastrowid
    
    # Return a dict compatible with the old API (includes person field)
    return {
        "id": transaction["id"],
        "person": name,
        "amount": amount,
        "direction": direction,
        "status": "open",
        "user_id": uid,
        "debtor_id": debtor["id"],
        "transaction_type": "debt",
    }


def mark_debt_paid(
    user_id: int,
    person: str,
    *,
    db_path=None,
) -> Optional[dict[str, Any]]:
    """Record a payment for a debtor using the new system.
    
    This function:
    1. Finds the debtor by exact name match
    2. Adds a payment transaction to reduce their balance
    3. Also marks the oldest open debt in the legacy table as paid for backward compatibility
    
    Returns a dict with person, amount, direction, and other fields for backward compatibility.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    name = (person or "").strip()
    
    # Find debtor by exact name match
    debtor = _get_or_create_debtor(uid, name, db_path=db_path)
    
    # Get current balance to determine direction
    balance = get_debtor_balance(debtor["id"], db_path=db_path)
    
    # Determine which direction to record payment for
    # If they owe us (total_owed_to_me > 0), payment reduces that
    # If we owe them (total_i_owe > 0), payment reduces that
    if balance["total_owed_to_me"] > 0:
        direction = "owed_to_me"
    elif balance["total_i_owe"] > 0:
        direction = "i_owe"
    else:
        # No existing debt - can't record payment
        return None
    
    # Get the amount of the oldest open debt to know how much to record
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM debts
                    WHERE user_id = %s AND status = 'open' AND LOWER(person) = LOWER(%s)
                    ORDER BY id ASC
                    LIMIT 1;
                    """,
                    (uid, name),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                amount = row["amount"]
                
                # Mark legacy debt as paid
                cur.execute(
                    """
                    UPDATE debts
                    SET status = 'paid', paid_at = NOW()
                    WHERE id = %s;
                    """,
                    (row["id"],),
                )
        else:
            row = conn.execute(
                """
                SELECT * FROM debts
                WHERE user_id = ? AND status = 'open' AND LOWER(person) = LOWER(?)
                ORDER BY id ASC
                LIMIT 1;
                """,
                (uid, name),
            ).fetchone()
            if row is None:
                return None
            amount = row["amount"]
            
            # Mark legacy debt as paid
            conn.execute(
                """
                UPDATE debts
                SET status = 'paid', paid_at = datetime('now')
                WHERE id = ?;
                """,
                (row["id"],),
            )
    
    # Add payment transaction to new system
    transaction = add_debt_transaction(
        uid, debtor["id"], amount, direction, transaction_type="payment", db_path=db_path
    )
    
    # Return a dict compatible with the old API
    return {
        "id": transaction["id"],
        "person": name,
        "amount": amount,
        "direction": direction,
        "status": "paid",
        "user_id": uid,
        "debtor_id": debtor["id"],
        "transaction_type": "payment",
    }


def list_open_debts(
    user_id: int,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """All open ("unpaid") debts for a user, using the new debtor system.
    
    Returns a list of debt dicts with debtor info and current balance.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT d.id as debtor_id, d.canonical_name, d.original_name, d.created_at,
                           dt.id as transaction_id, dt.amount, dt.direction, dt.transaction_type, dt.notes, dt.created_at as trans_created_at
                    FROM debtors d
                    LEFT JOIN debt_transactions dt ON d.id = dt.debtor_id
                    WHERE d.user_id = %s
                    ORDER BY d.created_at DESC, dt.created_at DESC;
                    """,
                    (uid,),
                )
                rows = [row_to_dict(row) for row in cur.fetchall()]
        else:
            rows = conn.execute(
                """
                SELECT d.id as debtor_id, d.canonical_name, d.original_name, d.created_at,
                       dt.id as transaction_id, dt.amount, dt.direction, dt.transaction_type, dt.notes, dt.created_at as trans_created_at
                FROM debtors d
                LEFT JOIN debt_transactions dt ON d.id = dt.debtor_id
                WHERE d.user_id = ?
                ORDER BY d.created_at DESC, dt.created_at DESC;
                """,
                (uid,),
            ).fetchall()
            rows = [row_to_dict(r) for r in rows]
    
    # Group by debtor and calculate balances
    debtor_balances = {}
    for row in rows:
        debtor_id = row["debtor_id"]
        if debtor_id not in debtor_balances:
            debtor_balances[debtor_id] = {
                "person": row["canonical_name"],
                "total_owed_to_me": 0,
                "total_i_owe": 0,
                "transactions": [],
            }
        
        if row["transaction_id"]:  # Has transactions
            debtor_balances[debtor_id]["transactions"].append(row)
            if row["transaction_type"] == "debt":
                if row["direction"] == "owed_to_me":
                    debtor_balances[debtor_id]["total_owed_to_me"] += int(row["amount"])
                else:
                    debtor_balances[debtor_id]["total_i_owe"] += int(row["amount"])
            elif row["transaction_type"] == "payment":
                if row["direction"] == "owed_to_me":
                    debtor_balances[debtor_id]["total_owed_to_me"] -= int(row["amount"])
                else:
                    debtor_balances[debtor_id]["total_i_owe"] -= int(row["amount"])
    
    # Filter to only debtors with outstanding balances
    open_debts = []
    for debtor_id, info in debtor_balances.items():
        net_owed_to_me = info["total_owed_to_me"]
        net_i_owe = info["total_i_owe"]
        
        if net_owed_to_me > 0:
            open_debts.append({
                "person": info["person"],
                "amount": net_owed_to_me,
                "direction": "owed_to_me",
                "status": "open",
            })
        elif net_i_owe > 0:
            open_debts.append({
                "person": info["person"],
                "amount": net_i_owe,
                "direction": "i_owe",
                "status": "open",
            })
    
    return open_debts


def get_summary(
    user_id: int,
    days: int = 7,
    *,
    db_path=None,
) -> dict[str, Any]:
    """Aggregate summary over the last `days` days, ignoring voided rows.

    Returns dict keys:
      - total_sales    int
      - total_expenses int
      - profit         int  (sales - expenses, can be negative)
      - top_sale_item       Optional[dict] with keys {item, total_amount, count}
      - top_expense_item    Optional[dict] same shape
      - total_owed_to_me    int  (open debts where the person owes the trader)
      - total_i_owe         int  (open debts where the trader owes the person)

    Debt totals are current balances (no `days` window); only open debts count.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    days = max(1, int(days))

    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Date function differs: PostgreSQL uses NOW() - INTERVAL vs SQLite datetime('now', '-')
                days_interval = f"{days} days"
                cur.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END), 0)    AS total_sales,
                        COALESCE(SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END), 0) AS total_expenses
                    FROM entries
                    WHERE user_id = %s
                      AND status = 'active'
                      AND created_at >= NOW() - INTERVAL %s;
                    """,
                    (uid, days_interval),
                )
                totals = cur.fetchone()
                total_sales = int(totals["total_sales"])
                total_expenses = int(totals["total_expenses"])

                cur.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE 
                            WHEN direction = 'owed_to_me' AND transaction_type = 'debt' THEN amount 
                            ELSE 0 
                        END), 0) AS debt_owed_to_me,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'owed_to_me' AND transaction_type = 'payment' THEN amount 
                            ELSE 0 
                        END), 0) AS payment_owed_to_me,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'i_owe' AND transaction_type = 'debt' THEN amount 
                            ELSE 0 
                        END), 0) AS debt_i_owe,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'i_owe' AND transaction_type = 'payment' THEN amount 
                            ELSE 0 
                        END), 0) AS payment_i_owe
                    FROM debt_transactions
                    WHERE debtor_id IN (
                        SELECT id FROM debtors WHERE user_id = %s
                    );
                    """,
                    (uid,),
                )
                debt_totals = cur.fetchone()
                total_owed_to_me = int(debt_totals["debt_owed_to_me"]) - int(debt_totals["payment_owed_to_me"])
                total_i_owe = int(debt_totals["debt_i_owe"]) - int(debt_totals["payment_i_owe"])

                def _top_item(row_type: str) -> Optional[dict[str, Any]]:
                    days_interval = f"{days} days"
                    cur.execute(
                        """
                        SELECT item,
                               SUM(amount)             AS total_amount,
                               COUNT(*)                AS count
                        FROM entries
                        WHERE user_id = %s
                          AND status = 'active'
                          AND type = %s
                          AND created_at >= NOW() - INTERVAL %s
                        GROUP BY item
                        ORDER BY SUM(amount) DESC, item ASC
                        LIMIT 1;
                        """,
                        (uid, row_type, days_interval),
                    )
                    row = cur.fetchone()
                    if row is None or row["total_amount"] is None:
                        return None
                    return {
                        "item": row["item"],
                        "total_amount": int(row["total_amount"]),
                        "count": int(row["count"]),
                    }

                return {
                    "total_sales": total_sales,
                    "total_expenses": total_expenses,
                    "profit": total_sales - total_expenses,
                    "top_sale_item": _top_item("sale"),
                    "top_expense_item": _top_item("expense"),
                    "total_owed_to_me": total_owed_to_me,
                    "total_i_owe": total_i_owe,
                }
        else:
            totals = conn.execute(
                f"""
                SELECT
                    COALESCE(SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END), 0)    AS total_sales,
                    COALESCE(SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END), 0) AS total_expenses
                FROM entries
                WHERE user_id = ?
                  AND status = 'active'
                  AND datetime(created_at) >= datetime('now', ?);
                """,
                (uid, f"-{days} days"),
            ).fetchone()
            total_sales = int(totals["total_sales"])
            total_expenses = int(totals["total_expenses"])

            debt_totals = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE 
                        WHEN direction = 'owed_to_me' AND transaction_type = 'debt' THEN amount 
                        ELSE 0 
                    END), 0) AS debt_owed_to_me,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'owed_to_me' AND transaction_type = 'payment' THEN amount 
                        ELSE 0 
                    END), 0) AS payment_owed_to_me,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'i_owe' AND transaction_type = 'debt' THEN amount 
                        ELSE 0 
                    END), 0) AS debt_i_owe,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'i_owe' AND transaction_type = 'payment' THEN amount 
                        ELSE 0 
                    END), 0) AS payment_i_owe
                FROM debt_transactions
                WHERE debtor_id IN (
                    SELECT id FROM debtors WHERE user_id = ?
                );
                """,
                (uid,),
            ).fetchone()
            
            total_owed_to_me = int(debt_totals["debt_owed_to_me"]) - int(debt_totals["payment_owed_to_me"])
            total_i_owe = int(debt_totals["debt_i_owe"]) - int(debt_totals["payment_i_owe"])

            def _top_item(row_type: str) -> Optional[dict[str, Any]]:
                row = conn.execute(
                    f"""
                    SELECT item,
                           SUM(amount)             AS total_amount,
                           COUNT(*)                AS count
                    FROM entries
                    WHERE user_id = ?
                      AND status = 'active'
                      AND type = ?
                      AND datetime(created_at) >= datetime('now', ?)
                    GROUP BY item
                    ORDER BY SUM(amount) DESC, item ASC
                    LIMIT 1;
                    """,
                    (uid, row_type, f"-{days} days"),
                ).fetchone()
                if row is None or row["total_amount"] is None:
                    return None
                return {
                    "item": row["item"],
                    "total_amount": int(row["total_amount"]),
                    "count": int(row["count"]),
                }

            return {
                "total_sales": total_sales,
                "total_expenses": total_expenses,
                "profit": total_sales - total_expenses,
                "top_sale_item": _top_item("sale"),
                "top_expense_item": _top_item("expense"),
                "total_owed_to_me": total_owed_to_me,
                "total_i_owe": total_i_owe,
            }


# ---------------------------------------------------------------------------
# Persistent ledger link (Excel export via phone number)
# ---------------------------------------------------------------------------

def get_user_by_id(
    user_id: int,
    *,
    db_path=None,
) -> Optional[dict[str, Any]]:
    """Return the full user row (including phone_or_name) for a user_id, or None."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    return query_one(
        db_path,
        "SELECT * FROM users WHERE id = ? LIMIT 1;",
        (int(user_id),),
    )


def list_active_entries_chronological(
    user_id: int,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """Active entries only, oldest first (needed for running-balance xlsx)."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM entries
                    WHERE user_id = %s AND status = 'active'
                    ORDER BY created_at ASC, id ASC;
                    """,
                    (uid,),
                )
                return [row_to_dict(row) for row in cur.fetchall()]
        else:
            rows = conn.execute(
                """
                SELECT * FROM entries
                WHERE user_id = ? AND status = 'active'
                ORDER BY datetime(created_at) ASC, id ASC;
                """,
                (uid,),
            ).fetchall()
            return [row_to_dict(r) for r in rows]


def generate_xlsx_ledger_bytes(
    user_id: int,
    *,
    db_path=None,
    phone_label: Optional[str] = None,
) -> bytes:
    """Build a fresh in-memory .xlsx workbook for ``user_id``.

    Columns (one row per active entry, chronological order with a running
    running balance where sales add and expenses subtract):
      - Date        (created_at, formatted)
      - Item        (the item name, e.g. "Rice")
      - Amount      (integer naira, sale is positive, expense is negative)
      - Type        (Sale / Expense)
      - Balance     (running balance after this row)

    The workbook is NEVER cached on disk: callers stream it out fresh on
    every request so the file is always up to date.
    """
    # openpyxl is a declared dep; import here so `import app.ledger` stays
    # cheap and unit tests that don't touch Excel still run without it.
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    entries = list_active_entries_chronological(user_id, db_path=db_path)

    wb = Workbook()
    ws = wb.active
    ws.title = "Ledger"

    # ---- Header row ----
    headers = ["Date", "Item", "Amount (₦)", "Type", "Balance (₦)"]
    ws.append(headers)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="0C3B2E")
    center = Alignment(horizontal="center")
    for col_idx, _ in enumerate(headers, 1):
        c = ws.cell(row=1, column=col_idx)
        c.font = header_font
        c.fill = header_fill
        c.alignment = center

    # ---- Data rows ----
    running = 0
    gold_fill = PatternFill("solid", fgColor="F7D27A")
    for entry in entries:
        amount_int = int(entry["amount"])
        signed = amount_int if entry["type"] == "sale" else -amount_int
        running += signed
        # Format: date (drop the seconds for readability; keep original precision)
        raw_date = entry.get("created_at") or ""
        ws.append([
            raw_date,
            entry.get("item", ""),
            signed,
            "Sale" if entry["type"] == "sale" else "Expense",
            running,
        ])

    # ---- Summary footer (blank separator + totals) ----
    summary_start = ws.max_row + 2
    ws.cell(row=summary_start, column=1, value="Summary").font = Font(bold=True)
    sales_total = sum(
        int(e["amount"]) for e in entries if e["type"] == "sale"
    )
    expense_total = sum(
        int(e["amount"]) for e in entries if e["type"] == "expense"
    )
    ws.cell(row=summary_start + 1, column=1, value="Total Sales")
    ws.cell(row=summary_start + 1, column=3, value=sales_total).font = Font(bold=True, color="17705A")
    ws.cell(row=summary_start + 2, column=1, value="Total Expenses")
    ws.cell(row=summary_start + 2, column=3, value=expense_total).font = Font(bold=True, color="B03A2E")
    ws.cell(row=summary_start + 3, column=1, value="Net Profit / (Loss)")
    ws.cell(row=summary_start + 3, column=3, value=sales_total - expense_total).font = Font(bold=True)
    ws.cell(row=summary_start + 3, column=3).fill = gold_fill

    # ---- Column widths ----
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 18

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_ledger_path(phone_or_name: str) -> str:
    """Return the URL path ``/ledger/{phone}.xlsx`` (URL-encoded safe)."""
    cleaned = (phone_or_name or "").strip()
    return f"/ledger/{quote(cleaned, safe='')}.xlsx"


# ---------------------------------------------------------------------------
# Insight functions
# ---------------------------------------------------------------------------

def get_stock_levels(
    user_id: int,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """Calculate stock levels for items with both buy and sell entries.
    
    For each item that has appeared in both "buy" (expense) and "sell" (sale) entries,
    calculate: total quantity bought minus total quantity sold. Only uses entries
    where quantity was captured (non-null). Skips items with no quantity data.
    
    Returns a list of dicts with keys: {item, quantity_remaining}
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Get items with both buy and sell entries that have quantity data
                cur.execute("""
                    SELECT item,
                           COALESCE(SUM(CASE WHEN type = 'expense' THEN quantity ELSE 0 END), 0) AS total_bought,
                           COALESCE(SUM(CASE WHEN type = 'sale' THEN quantity ELSE 0 END), 0) AS total_sold
                    FROM entries
                    WHERE user_id = %s
                      AND status = 'active'
                      AND quantity IS NOT NULL
                    GROUP BY item
                    HAVING SUM(CASE WHEN type = 'expense' THEN 1 ELSE 0 END) > 0
                       AND SUM(CASE WHEN type = 'sale' THEN 1 ELSE 0 END) > 0
                    ORDER BY item ASC;
                """, (uid,))
                
                stock_levels = []
                for row in cur.fetchall():
                    bought = float(row["total_bought"]) if row["total_bought"] else 0
                    sold = float(row["total_sold"]) if row["total_sold"] else 0
                    remaining = bought - sold
                    stock_levels.append({
                        "item": row["item"],
                        "quantity_remaining": round(remaining, 2)
                    })
                return stock_levels
        else:
            # SQLite version
            rows = conn.execute("""
                SELECT item,
                       COALESCE(SUM(CASE WHEN type = 'expense' THEN quantity ELSE 0 END), 0) AS total_bought,
                       COALESCE(SUM(CASE WHEN type = 'sale' THEN quantity ELSE 0 END), 0) AS total_sold
                FROM entries
                WHERE user_id = ?
                  AND status = 'active'
                  AND quantity IS NOT NULL
                GROUP BY item
                HAVING SUM(CASE WHEN type = 'expense' THEN 1 ELSE 0 END) > 0
                   AND SUM(CASE WHEN type = 'sale' THEN 1 ELSE 0 END) > 0
                ORDER BY item ASC;
            """, (uid,)).fetchall()
            
            stock_levels = []
            for row in rows:
                bought = float(row["total_bought"]) if row["total_bought"] else 0
                sold = float(row["total_sold"]) if row["total_sold"] else 0
                remaining = bought - sold
                stock_levels.append({
                    "item": row["item"],
                    "quantity_remaining": round(remaining, 2)
                })
            return stock_levels


def get_top_items(
    user_id: int,
    metric: str = "profit",
    period_days: int = 30,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """Return items ranked by profit or transaction volume over a period.
    
    Args:
        user_id: The user's ID
        metric: "profit" (sales minus cost using average buy price) or "volume" (total transactions)
        period_days: Number of days to look back (default 30)
    
    Returns a list of dicts with keys: {item, total_profit, total_volume, transaction_count}
    Ranked in descending order by the specified metric.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    period_days = max(1, int(period_days))
    
    if metric not in ("profit", "volume"):
        raise ValueError(f"metric must be 'profit' or 'volume', got {metric!r}")
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                # Get item performance data
                days_interval = f"{period_days} days"
                cur.execute("""
                    SELECT item,
                           SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) AS total_sales,
                           SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) AS total_cost,
                           COUNT(*) AS transaction_count,
                           SUM(amount) AS total_volume
                    FROM entries
                    WHERE user_id = %s
                      AND status = 'active'
                      AND created_at >= NOW() - INTERVAL %s
                    GROUP BY item
                    HAVING SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) > 0
                    ORDER BY item ASC;
                """, (uid, days_interval))
                
                items = []
                for row in cur.fetchall():
                    total_sales = int(row["total_sales"]) if row["total_sales"] else 0
                    total_cost = int(row["total_cost"]) if row["total_cost"] else 0
                    total_profit = total_sales - total_cost
                    total_volume = int(row["total_volume"]) if row["total_volume"] else 0
                    transaction_count = int(row["transaction_count"]) if row["transaction_count"] else 0
                    
                    items.append({
                        "item": row["item"],
                        "total_profit": total_profit,
                        "total_volume": total_volume,
                        "transaction_count": transaction_count
                    })
                
                # Sort by the requested metric
                if metric == "profit":
                    items.sort(key=lambda x: x["total_profit"], reverse=True)
                else:  # volume
                    items.sort(key=lambda x: x["total_volume"], reverse=True)
                
                return items
        else:
            # SQLite version
            rows = conn.execute("""
                SELECT item,
                       SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) AS total_sales,
                       SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) AS total_cost,
                       COUNT(*) AS transaction_count,
                       SUM(amount) AS total_volume
                FROM entries
                WHERE user_id = ?
                  AND status = 'active'
                  AND datetime(created_at) >= datetime('now', ?)
                GROUP BY item
                HAVING SUM(CASE WHEN type = 'sale' THEN amount ELSE 0 END) > 0
                ORDER BY item ASC;
            """, (uid, f"-{period_days} days")).fetchall()
            
            items = []
            for row in rows:
                total_sales = int(row["total_sales"]) if row["total_sales"] else 0
                total_cost = int(row["total_cost"]) if row["total_cost"] else 0
                total_profit = total_sales - total_cost
                total_volume = int(row["total_volume"]) if row["total_volume"] else 0
                transaction_count = int(row["transaction_count"]) if row["transaction_count"] else 0
                
                items.append({
                    "item": row["item"],
                    "total_profit": total_profit,
                    "total_volume": total_volume,
                    "transaction_count": transaction_count
                })
            
            # Sort by the requested metric
            if metric == "profit":
                items.sort(key=lambda x: x["total_profit"], reverse=True)
            else:  # volume
                items.sort(key=lambda x: x["total_volume"], reverse=True)
            
            return items


def get_expense_breakdown(
    user_id: int,
    period_days: int = 30,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """Group expense entries by item/category and return totals for the period.
    
    Args:
        user_id: The user's ID
        period_days: Number of days to look back (default 30)
    
    Returns a list of dicts with keys: {category, total_amount, entry_count}
    Sorted by total amount in descending order.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    period_days = max(1, int(period_days))
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                days_interval = f"{period_days} days"
                cur.execute("""
                    SELECT item AS category,
                           SUM(amount) AS total_amount,
                           COUNT(*) AS entry_count
                    FROM entries
                    WHERE user_id = %s
                      AND type = 'expense'
                      AND status = 'active'
                      AND created_at >= NOW() - INTERVAL %s
                    GROUP BY item
                    ORDER BY total_amount DESC;
                """, (uid, days_interval))
                
                breakdown = []
                for row in cur.fetchall():
                    breakdown.append({
                        "category": row["category"],
                        "total_amount": int(row["total_amount"]) if row["total_amount"] else 0,
                        "entry_count": int(row["entry_count"]) if row["entry_count"] else 0
                    })
                return breakdown
        else:
            # SQLite version
            rows = conn.execute("""
                SELECT item AS category,
                       SUM(amount) AS total_amount,
                       COUNT(*) AS entry_count
                FROM entries
                WHERE user_id = ?
                  AND type = 'expense'
                  AND status = 'active'
                  AND datetime(created_at) >= datetime('now', ?)
                GROUP BY item
                ORDER BY total_amount DESC;
            """, (uid, f"-{period_days} days")).fetchall()
            
            breakdown = []
            for row in rows:
                breakdown.append({
                    "category": row["category"],
                    "total_amount": int(row["total_amount"]) if row["total_amount"] else 0,
                    "entry_count": int(row["entry_count"]) if row["entry_count"] else 0
                })
            return breakdown


# Stock threshold for low stock warnings (configurable)
STOCK_THRESHOLD = 3  # Default threshold for low stock warnings


def check_low_stock(user_id: int, item: str, *, db_path=None) -> Optional[str]:
    """Check if an item's stock is below threshold and return a warning message.
    
    Returns a warning message if stock is low, None otherwise.
    """
    stock_levels = get_stock_levels(user_id, db_path=db_path)
    for stock in stock_levels:
        if stock["item"].lower() == item.lower():
            if stock["quantity_remaining"] <= STOCK_THRESHOLD:
                return f"You have only {stock['quantity_remaining']} {item} left."
            break
    return None


# ---------------------------------------------------------------------------
# Debtor Identity and Accumulation System
# ---------------------------------------------------------------------------

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings for fuzzy name matching."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _find_similar_debtors(
    user_id: int,
    name: str,
    *,
    db_path=None,
    threshold: int = 2
) -> list[dict[str, Any]]:
    """Find debtors with similar names (within Levenshtein distance threshold).
    
    Returns a list of debtor dicts with similar names, sorted by similarity.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    name_normalized = name.strip().lower()
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, canonical_name, original_name, created_at
                    FROM debtors
                    WHERE user_id = %s;
                    """,
                    (uid,),
                )
                debtors = [row_to_dict(row) for row in cur.fetchall()]
        else:
            rows = conn.execute(
                """
                SELECT id, canonical_name, original_name, created_at
                FROM debtors
                WHERE user_id = ?;
                """,
                (uid,),
            ).fetchall()
            debtors = [row_to_dict(r) for r in rows]
    
    # Filter by similarity threshold
    similar = []
    for debtor in debtors:
        distance = _levenshtein_distance(name_normalized, debtor["canonical_name"].lower())
        if distance <= threshold and distance > 0:  # >0 to exclude exact matches
            similar.append((distance, debtor))
    
    # Sort by distance (closest match first)
    similar.sort(key=lambda x: x[0])
    return [debtor for _, debtor in similar]


def _get_or_create_debtor(
    user_id: int,
    name: str,
    *,
    db_path=None,
) -> dict[str, Any]:
    """Get existing debtor by exact name match, or create a new one.
    
    Returns the debtor dict (with id, canonical_name, original_name, etc.).
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    name_normalized = name.strip()
    
    with get_connection(db_path) as conn:
        # Try to find exact match first (case-insensitive)
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM debtors
                    WHERE user_id = %s AND LOWER(canonical_name) = LOWER(%s)
                    LIMIT 1;
                    """,
                    (uid, name_normalized),
                )
                existing = cur.fetchone()
                if existing:
                    return row_to_dict(existing)
                
                # Create new debtor
                cur.execute(
                    """
                    INSERT INTO debtors (user_id, canonical_name, original_name)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """,
                    (uid, name_normalized, name_normalized),
                )
                new_id = cur.fetchone()["id"]
                cur.execute(
                    "SELECT * FROM debtors WHERE id = %s;", (new_id,)
                )
                return row_to_dict(cur.fetchone())
        else:
            row = conn.execute(
                """
                SELECT * FROM debtors
                WHERE user_id = ? AND LOWER(canonical_name) = LOWER(?)
                LIMIT 1;
                """,
                (uid, name_normalized),
            ).fetchone()
            if row:
                return row_to_dict(row)
            
            # Create new debtor
            cur = conn.execute(
                """
                INSERT INTO debtors (user_id, canonical_name, original_name)
                VALUES (?, ?, ?);
                """,
                (uid, name_normalized, name_normalized),
            )
            new_row = conn.execute(
                "SELECT * FROM debtors WHERE id = ?;", (cur.lastrowid,)
            ).fetchone()
            return row_to_dict(new_row)


def add_debt_transaction(
    user_id: int,
    debtor_id: int,
    amount: int,
    direction: str,
    transaction_type: str = "debt",
    notes: str = "",
    *,
    db_path=None,
) -> dict[str, Any]:
    """Add a debt transaction to an existing debtor.
    
    Args:
        user_id: The user's ID
        debtor_id: The debtor's ID from the debtors table
        amount: The amount in naira
        direction: "owed_to_me" or "i_owe"
        transaction_type: "debt" (new debt) or "payment" (payment made/received)
        notes: Optional notes about the transaction
    
    Returns the inserted transaction row.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if direction not in DEBT_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DEBT_DIRECTIONS}, got {direction!r}"
        )
    if transaction_type not in ("debt", "payment"):
        raise ValueError(
            f"transaction_type must be 'debt' or 'payment', got {transaction_type!r}"
        )
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO debt_transactions (debtor_id, amount, direction, transaction_type, notes)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (debtor_id, int(amount), direction, transaction_type, notes),
                )
                new_id = cur.fetchone()["id"]
                cur.execute(
                    "SELECT * FROM debt_transactions WHERE id = %s;", (new_id,)
                )
                return row_to_dict(cur.fetchone())
        else:
            cur = conn.execute(
                """
                INSERT INTO debt_transactions (debtor_id, amount, direction, transaction_type, notes)
                VALUES (?, ?, ?, ?, ?);
                """,
                (debtor_id, int(amount), direction, transaction_type, notes),
            )
            row = conn.execute(
                "SELECT * FROM debt_transactions WHERE id = ?;", (cur.lastrowid,)
            ).fetchone()
            return row_to_dict(row)


def get_debtor_balance(
    debtor_id: int,
    *,
    db_path=None,
) -> dict[str, int]:
    """Calculate the current balance for a debtor.
    
    Returns a dict with:
        - total_owed_to_me: sum of "owed_to_me" debts minus payments
        - total_i_owe: sum of "i_owe" debts minus payments
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    with get_connection(db_path) as conn:
        if USE_POSTGRES:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE 
                            WHEN direction = 'owed_to_me' AND transaction_type = 'debt' THEN amount 
                            ELSE 0 
                        END), 0) AS debt_owed_to_me,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'owed_to_me' AND transaction_type = 'payment' THEN amount 
                            ELSE 0 
                        END), 0) AS payment_owed_to_me,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'i_owe' AND transaction_type = 'debt' THEN amount 
                            ELSE 0 
                        END), 0) AS debt_i_owe,
                        COALESCE(SUM(CASE 
                            WHEN direction = 'i_owe' AND transaction_type = 'payment' THEN amount 
                            ELSE 0 
                        END), 0) AS payment_i_owe
                    FROM debt_transactions
                    WHERE debtor_id = %s;
                    """,
                    (debtor_id,),
                )
                row = cur.fetchone()
        else:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE 
                        WHEN direction = 'owed_to_me' AND transaction_type = 'debt' THEN amount 
                        ELSE 0 
                    END), 0) AS debt_owed_to_me,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'owed_to_me' AND transaction_type = 'payment' THEN amount 
                        ELSE 0 
                    END), 0) AS payment_owed_to_me,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'i_owe' AND transaction_type = 'debt' THEN amount 
                        ELSE 0 
                    END), 0) AS debt_i_owe,
                    COALESCE(SUM(CASE 
                        WHEN direction = 'i_owe' AND transaction_type = 'payment' THEN amount 
                        ELSE 0 
                    END), 0) AS payment_i_owe
                FROM debt_transactions
                WHERE debtor_id = ?;
                """,
                (debtor_id,),
            ).fetchone()
        
        debt_owed_to_me = int(row["debt_owed_to_me"]) if row["debt_owed_to_me"] else 0
        payment_owed_to_me = int(row["payment_owed_to_me"]) if row["payment_owed_to_me"] else 0
        debt_i_owe = int(row["debt_i_owe"]) if row["debt_i_owe"] else 0
        payment_i_owe = int(row["payment_i_owe"]) if row["payment_i_owe"] else 0
        
        return {
            "total_owed_to_me": debt_owed_to_me - payment_owed_to_me,
            "total_i_owe": debt_i_owe - payment_i_owe,
        }

