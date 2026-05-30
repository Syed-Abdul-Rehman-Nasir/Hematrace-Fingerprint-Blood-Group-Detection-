# backend/database.py
"""SQLite persistence for patients and test results."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DATA_DIR, DB_PATH

os.makedirs(DATA_DIR, exist_ok=True)


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL CHECK (type IN ('normal', 'emerg')),
                name TEXT NOT NULL,
                age TEXT,
                gender TEXT,
                phone TEXT NOT NULL DEFAULT '—',
                notes TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                record_code TEXT NOT NULL UNIQUE,
                blood_group TEXT NOT NULL,
                confidence REAL,
                test_date TEXT NOT NULL,
                test_time TEXT NOT NULL,
                tested_by TEXT NOT NULL,
                tested_by_role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_tests_patient ON tests(patient_id);
            CREATE INDEX IF NOT EXISTS idx_patients_lookup
                ON patients(type, name COLLATE NOCASE, phone);
            """
        )


def _normalize_phone(phone):
    p = (phone or "").strip()
    return p if p else "—"


def find_or_create_patient(*, ptype, name, age, gender, phone, notes=None):
    name = (name or "").strip() or "Unknown"
    phone = _normalize_phone(phone)
    age = age if age is not None else "—"
    gender = gender if gender is not None else "—"
    now = _utc_now_iso()

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT id FROM patients
            WHERE type = ? AND name = ? COLLATE NOCASE AND phone = ?
            """,
            (ptype, name, phone),
        ).fetchone()

        if row:
            patient_id = row["id"]
            conn.execute(
                """
                UPDATE patients SET age = ?, gender = ?, notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (age, gender, notes, patient_id),
            )
            return patient_id

        cur = conn.execute(
            """
            INSERT INTO patients (type, name, age, gender, phone, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ptype, name, age, gender, phone, notes, now),
        )
        return cur.lastrowid


def _next_record_code(conn, ptype):
    prefix = "EMG" if ptype == "emerg" else "HT"
    row = conn.execute("SELECT COUNT(*) AS n FROM tests").fetchone()
    n = (row["n"] or 0) + 1
    return f"{prefix}-{n:04d}"


def create_test(
    *,
    ptype,
    name,
    age,
    gender,
    phone,
    notes=None,
    blood_group,
    confidence,
    test_date,
    test_time,
    tested_by,
    tested_by_role,
):
    patient_id = find_or_create_patient(
        ptype=ptype,
        name=name,
        age=age,
        gender=gender,
        phone=phone,
        notes=notes,
    )
    now = _utc_now_iso()

    with get_conn() as conn:
        record_code = _next_record_code(conn, ptype)
        cur = conn.execute(
            """
            INSERT INTO tests (
                patient_id, record_code, blood_group, confidence,
                test_date, test_time, tested_by, tested_by_role, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                record_code,
                blood_group,
                float(confidence) if confidence is not None else None,
                test_date,
                test_time,
                tested_by,
                tested_by_role,
                now,
            ),
        )
        test_id = cur.lastrowid

    return {
        "patient_id": patient_id,
        "test_id": test_id,
        "record_code": record_code,
    }


def _row_to_record(patient_row, test_row):
    """Shape expected by the Electron UI (`records` array)."""
    rec = {
        "id": test_row["record_code"],
        "patientId": patient_row["id"],
        "testId": test_row["id"],
        "name": patient_row["name"],
        "age": patient_row["age"] or "—",
        "gender": patient_row["gender"] or "—",
        "phone": patient_row["phone"] or "—",
        "type": patient_row["type"],
        "bg": test_row["blood_group"],
        "confidence": test_row["confidence"],
        "date": test_row["test_date"],
        "time": test_row["test_time"],
        "by": test_row["tested_by"],
        "byRole": test_row["tested_by_role"],
    }
    if patient_row["notes"]:
        rec["notes"] = patient_row["notes"]
    return rec


def list_all_records(ptype=None):
    query = """
        SELECT
            p.id AS patient_id, p.type, p.name, p.age, p.gender, p.phone, p.notes,
            t.id AS test_id, t.record_code, t.blood_group, t.confidence,
            t.test_date, t.test_time, t.tested_by, t.tested_by_role
        FROM tests t
        JOIN patients p ON p.id = t.patient_id
    """
    params = []
    if ptype:
        query += " WHERE p.type = ?"
        params.append(ptype)
    query += " ORDER BY t.id ASC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    records = []
    for row in rows:
        patient = {
            "id": row["patient_id"],
            "type": row["type"],
            "name": row["name"],
            "age": row["age"],
            "gender": row["gender"],
            "phone": row["phone"],
            "notes": row["notes"],
        }
        test = {
            "id": row["test_id"],
            "record_code": row["record_code"],
            "blood_group": row["blood_group"],
            "confidence": row["confidence"],
            "test_date": row["test_date"],
            "test_time": row["test_time"],
            "tested_by": row["tested_by"],
            "tested_by_role": row["tested_by_role"],
        }
        records.append(_row_to_record(patient, test))
    return records


def list_patients_summary(ptype):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                p.id, p.type, p.name, p.age, p.gender, p.phone, p.notes,
                COUNT(t.id) AS test_count,
                MAX(t.id) AS latest_test_id
            FROM patients p
            LEFT JOIN tests t ON t.patient_id = p.id
            WHERE p.type = ?
            GROUP BY p.id
            ORDER BY latest_test_id DESC, p.id DESC
            """,
            (ptype,),
        ).fetchall()

        patients = []
        for row in rows:
            latest = None
            if row["latest_test_id"]:
                latest = conn.execute(
                    """
                    SELECT record_code, blood_group, confidence, test_date, test_time,
                           tested_by, tested_by_role
                    FROM tests WHERE id = ?
                    """,
                    (row["latest_test_id"],),
                ).fetchone()

            patients.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "name": row["name"],
                    "age": row["age"] or "—",
                    "gender": row["gender"] or "—",
                    "phone": row["phone"] or "—",
                    "notes": row["notes"],
                    "testCount": row["test_count"] or 0,
                    "latest": dict(latest) if latest else None,
                }
            )
    return patients


def get_patient_tests(patient_id):
    with get_conn() as conn:
        patient = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if not patient:
            return None
        tests = conn.execute(
            """
            SELECT * FROM tests WHERE patient_id = ?
            ORDER BY id DESC
            """,
            (patient_id,),
        ).fetchall()

    return {
        "patient": dict(patient),
        "tests": [dict(t) for t in tests],
    }
