"""SQLite storage for students and submissions, shared by the web app and the worker."""
import json
import sqlite3
from contextlib import contextmanager

from app.settings import data_dir, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    email         TEXT,
    password_hash TEXT NOT NULL,
    is_baseline   INTEGER NOT NULL DEFAULT 0,   -- reference entry shown on leaderboards, never ranked
    is_staff      INTEGER NOT NULL DEFAULT 0,   -- may use the admin page; never ranked
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS submissions (
    id            INTEGER PRIMARY KEY,
    student_id    INTEGER NOT NULL REFERENCES students(id),
    track         TEXT NOT NULL,
    test_set      TEXT NOT NULL,                -- validation | hidden
    kind          TEXT NOT NULL,                -- zip | image
    base_image    TEXT,                         -- key in config base_images (zip submissions)
    note          TEXT,                         -- short description by the student
    upload_path   TEXT NOT NULL,
    upload_bytes  INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'queued',  -- queued | running | succeeded | failed | cancelled
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    wall_seconds  REAL,
    scores        TEXT,                         -- JSON from evaluate.py
    error         TEXT                          -- short reason shown to the student
);
CREATE INDEX IF NOT EXISTS idx_sub_status ON submissions(status);
CREATE INDEX IF NOT EXISTS idx_sub_student ON submissions(student_id, test_set);
"""


def path():
    return data_dir() / "nncv.sqlite"


@contextmanager
def connect():
    """A connection with rows as dicts; commits on success."""
    data_dir().mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path(), timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")     # web app and worker read and write at the same time
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with connect() as con:
        con.executescript(SCHEMA)
        columns = {r["name"] for r in con.execute("PRAGMA table_info(students)")}
        if "is_staff" not in columns:              # databases made before the staff role existed
            con.execute("ALTER TABLE students ADD COLUMN is_staff INTEGER NOT NULL DEFAULT 0")


def stamp() -> str:
    return now().isoformat(timespec="seconds")


def row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("scores"):
        d["scores"] = json.loads(d["scores"])
    return d


# ------------------------------------------------------------------ students
def students(include_inactive=False):
    with connect() as con:
        q = "SELECT * FROM students" + ("" if include_inactive else " WHERE active = 1") + " ORDER BY name COLLATE NOCASE"
        return [dict(r) for r in con.execute(q)]


def student_by_name(name):
    with connect() as con:
        r = con.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        return dict(r) if r else None


def student_by_id(sid):
    with connect() as con:
        r = con.execute("SELECT * FROM students WHERE id = ?", (sid,)).fetchone()
        return dict(r) if r else None


def upsert_student(name, email, password_hash, is_baseline=False, is_staff=False):
    with connect() as con:
        con.execute("""INSERT INTO students (name, email, password_hash, is_baseline, is_staff) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(name) DO UPDATE SET email = excluded.email, password_hash = excluded.password_hash,
                       is_baseline = excluded.is_baseline, is_staff = excluded.is_staff, active = 1""",
                    (name, email, password_hash, int(is_baseline), int(is_staff)))


def update_student(sid, **fields):
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE students SET {sets} WHERE id = ?", (*fields.values(), sid))


# ------------------------------------------------------------------ submissions
SUB_COLUMNS = "s.*, st.name AS student, st.is_baseline, st.is_staff"


def add_submission(student_id, track, test_set, kind, base_image, note, upload_path, upload_bytes) -> int:
    with connect() as con:
        cur = con.execute("""INSERT INTO submissions (student_id, track, test_set, kind, base_image, note,
                             upload_path, upload_bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                          (student_id, track, test_set, kind, base_image, note, str(upload_path), upload_bytes, stamp()))
        return cur.lastrowid


def submission(sub_id):
    with connect() as con:
        r = con.execute(f"SELECT {SUB_COLUMNS} FROM submissions s JOIN students st ON st.id = s.student_id WHERE s.id = ?",
                        (sub_id,)).fetchone()
        return row_to_dict(r) if r else None


def submissions(where="1=1", params=(), order="s.id DESC", limit=None):
    q = f"SELECT {SUB_COLUMNS} FROM submissions s JOIN students st ON st.id = s.student_id WHERE {where} ORDER BY {order}"
    if limit:
        q += f" LIMIT {int(limit)}"
    with connect() as con:
        return [row_to_dict(r) for r in con.execute(q, params)]


def update_submission(sub_id, **fields):
    if "scores" in fields and fields["scores"] is not None:
        fields["scores"] = json.dumps(fields["scores"])
    sets = ", ".join(f"{k} = ?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE submissions SET {sets} WHERE id = ?", (*fields.values(), sub_id))
