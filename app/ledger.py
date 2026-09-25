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

from app.db import DEFAULT_DB_PATH, get_connection, get_or_create_user, query_one, row_to_dict


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
    conn: sqlite3.Connection,
    user_id: int,
) -> Optional[sqlite3.Row]:
    """The most recent active entry for a user, or None."""
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
    """Record a debt. `direction` is "owed_to_me" (they owe the trader) or
    "i_owe" (the trader owes them). Returns the inserted row."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if direction not in DEBT_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {DEBT_DIRECTIONS}, got {direction!r}"
        )
    uid = _require_user(db_path, user_id)
    name = (person or "").strip() or "Unknown"
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO debts (user_id, person, amount, direction, status)
            VALUES (?, ?, ?, ?, 'open');
            """,
            (uid, name, int(amount), direction),
        )
        row = conn.execute(
            "SELECT * FROM debts WHERE id = ?;", (cur.lastrowid,)
        ).fetchone()
        return row_to_dict(row)


def mark_debt_paid(
    user_id: int,
    person: str,
    *,
    db_path=None,
) -> Optional[dict[str, Any]]:
    """Settle the OLDEST open debt for `person` (case-insensitive exact name).

    Returns the updated row, or None when no open debt matches.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    name = (person or "").strip()
    with get_connection(db_path) as conn:
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
        conn.execute(
            """
            UPDATE debts
            SET status = 'paid', paid_at = datetime('now')
            WHERE id = ?;
            """,
            (row["id"],),
        )
        updated = conn.execute(
            "SELECT * FROM debts WHERE id = ?;", (row["id"],)
        ).fetchone()
        return row_to_dict(updated)


def list_open_debts(
    user_id: int,
    *,
    db_path=None,
) -> list[dict[str, Any]]:
    """All open ("unpaid") debts for a user, newest first."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    uid = _require_user(db_path, user_id)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM debts
            WHERE user_id = ? AND status = 'open'
            ORDER BY created_at DESC, id DESC;
            """,
            (uid,),
        ).fetchall()
        return [row_to_dict(r) for r in rows]


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
                COALESCE(SUM(CASE WHEN direction = 'owed_to_me' THEN amount ELSE 0 END), 0) AS total_owed_to_me,
                COALESCE(SUM(CASE WHEN direction = 'i_owe'       THEN amount ELSE 0 END), 0) AS total_i_owe
            FROM debts
            WHERE user_id = ?
              AND status = 'open';
            """,
            (uid,),
        ).fetchone()

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
            "total_owed_to_me": int(debt_totals["total_owed_to_me"]),
            "total_i_owe": int(debt_totals["total_i_owe"]),
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

