"""
Ledger operations backed by app.db.

Three public functions mirror the requirements exactly:

  - add_entries(user_id, entries, transcript, audio_file)
  - list_entries(user_id, limit=50)
  - get_summary(user_id, days=7)

`get_summary` returns total sales, total expenses, profit and the top sale
item (by summed amount) and top expense item (by summed amount). Only
"active"-status entries contribute.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from app.db import DEFAULT_DB_PATH, get_connection, get_or_create_user, row_to_dict


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_entries(
    user_id: Optional[int],
    entries: Iterable[dict[str, Any]],
    transcript: str,
    audio_file: str = "",
    *,
    db_path=DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Insert a batch of parsed entries for a user, attach transcript+audio_file metadata
    on each row. Returns the freshly inserted rows (with generated ids)."""
    entries = list(entries)
    if not entries:
        return []
    uid = _require_user(db_path, user_id)

    with get_connection(db_path) as conn:
        insert_sql = """
            INSERT INTO entries (user_id, item, quantity, amount, type,
                             transcript, audio_file, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
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
                )
            )
        conn.executemany(insert_sql, rows)
        # Pull back the inserted rows in insertion order. SQLite's lastrowid after
        # executemany returns the id of the FIRST inserted row; the others are
        # contiguous integers, so we can slice by the count and match the original
        # entries order reliably.
        first_id = conn.execute(
            "SELECT last_insert_rowid() AS id FROM entries LIMIT 1;"
        ).fetchone()["id"]
        last_id = first_id + len(rows) - 1
        fetched = conn.execute(
            f"""
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
    db_path=DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return the most recent `limit` entries for a user (all statuses)."""
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
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


def get_summary(
    user_id: int,
    days: int = 7,
    *,
    db_path=DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Aggregate summary over the last `days` days, ignoring voided rows.

    Returns dict keys:
      - total_sales    int
      - total_expenses int
      - profit         int  (sales - expenses, can be negative)
      - top_sale_item       Optional[dict] with keys {item, total_amount, count}
      - top_expense_item    Optional[dict] same shape
    """
    uid = _require_user(db_path, user_id)
    days = max(1, int(days))

    with get_connection(db_path) as conn:
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
        }
