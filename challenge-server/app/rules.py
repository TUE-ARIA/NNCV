"""Challenge rules: who may submit what, the queue order, and the leaderboards."""
from pathlib import Path

from app import db
from app.settings import config, now, parse_time, week_start

PENDING = ("queued", "running")


# ------------------------------------------------------------------ submitting
def submission_check(student: dict, track: str, test_set: str) -> str | None:
    """Reason why this student cannot submit now, or None if allowed."""
    cfg = config()
    sub_cfg, set_cfg = cfg["submissions"], cfg["test_sets"].get(test_set)
    if track not in cfg["tracks"]:
        return "Unknown track."
    if not set_cfg or not set_cfg["enabled"]:
        return f"Submissions to the {set_cfg['label'] if set_cfg else test_set} set are not open."
    if not sub_cfg["open"]:
        return "Submissions are closed at the moment."
    deadline = parse_time(sub_cfg.get("deadline"))
    if deadline and now() > deadline:
        return f"The deadline ({deadline:%d %B %Y, %H:%M}) has passed."
    if not student["active"]:
        return "This account is not active."

    pending = db.submissions("s.student_id = ? AND s.status IN ('queued', 'running')", (student["id"],))
    if len(pending) >= sub_cfg["max_pending_per_student"]:
        return "You already have a submission waiting or running. Submit again when it has finished."

    used, limit = quota_used(student["id"], track, test_set), set_cfg["succeeded_per_week"]
    if not (student["is_baseline"] or student["is_staff"]) and used >= limit:
        return f"You have used all {limit} successful submissions to this set for this week. The quota resets on Monday."
    return None


def quota_used(student_id: int, track: str, test_set: str) -> int:
    """Succeeded this week, plus everything still waiting or running."""
    where = "s.student_id = ? AND s.test_set = ? AND (s.status IN ('queued', 'running') OR (s.status = 'succeeded' AND s.created_at >= ?))"
    params = [student_id, test_set, week_start().isoformat(timespec="seconds")]
    if config().get("quota_per_track", True):
        where += " AND s.track = ?"
        params.append(track)
    return len(db.submissions(where, tuple(params)))


# ------------------------------------------------------------------ queue
def queue() -> list[dict]:
    """Running job first, then waiting jobs in the order the worker will take them."""
    running = db.submissions("s.status = 'running'")
    waiting = db.submissions("s.status = 'queued'", order="s.id")
    return running + fair_order(waiting)


def fair_order(waiting: list[dict]) -> list[dict]:
    """Round-robin over students: whoever was served longest ago goes first."""
    last_start = {}
    for row in db.submissions("s.started_at IS NOT NULL", order="s.started_at"):
        last_start[row["student_id"]] = row["started_at"]
    return sorted(waiting, key=lambda s: (last_start.get(s["student_id"], ""), s["id"]))


def next_job() -> dict | None:
    order = fair_order(db.submissions("s.status = 'queued'", order="s.id"))
    return order[0] if order else None


def cancel(sub: dict) -> str | None:
    """Cancel a waiting submission; returns the reason if that is not possible."""
    if sub["status"] != "queued":
        return f"Submission {sub['id']} is {sub['status']}; only waiting submissions can be cancelled."
    db.update_submission(sub["id"], status="cancelled", finished_at=db.stamp(), error="Cancelled by staff.")
    return None


def requeue(sub: dict) -> str | None:
    """Put a finished submission back in the queue; returns the reason if that is not possible."""
    if sub["status"] in PENDING:
        return f"Submission {sub['id']} is already {sub['status']}."
    if not Path(sub["upload_path"]).exists():
        return "The upload of this submission no longer exists."
    db.update_submission(sub["id"], status="queued", started_at=None, finished_at=None, scores=None, error=None)
    return None


# ------------------------------------------------------------------ ranking
def ranking(track: str, scores: dict | None) -> tuple[float | None, str | None]:
    """(value to rank on, reason it is not ranked)."""
    t = config()["tracks"][track]
    if not scores or t["ranked_on"] not in scores or scores[t["ranked_on"]] is None:
        return None, "no score"
    if "min_fps" in t and scores.get("fps", 0) < t["min_fps"]:
        return scores[t["ranked_on"]], f"below {t['min_fps']} FPS"
    if "mask_dice_floor" in t:
        real = scores.get("mask_dice_real") or 0
        if real and scores.get("mask_dice", 0) < t["mask_dice_floor"] * real:
            return scores[t["ranked_on"]], f"mask Dice below {t['mask_dice_floor']:.0%} of real frames"
    return scores[t["ranked_on"]], None


def leaderboard(track: str, test_set: str, student: str | None = None) -> list[dict]:
    """Best succeeded submission per student (or all of one student), ranked rows first."""
    t = config()["tracks"][track]
    where, params = "s.track = ? AND s.test_set = ? AND s.status = 'succeeded'", [track, test_set]
    if student:
        where += " AND st.name = ?"
        params.append(student)
    rows = db.submissions(where, tuple(params), order="s.id")

    sign = 1 if t["higher_is_better"] else -1
    entries = []
    for r in rows:
        r["value"], r["unranked"] = ranking(track, r["scores"])
        r["reference"] = bool(r["is_baseline"] or r["is_staff"])   # shown for comparison, never ranked
        entries.append(r)
    if not student:                                   # keep each student's best entry
        best = {}
        for r in entries:
            key = r["student_id"]
            if key not in best or _better(r, best[key], sign):
                best[key] = r
        entries = list(best.values())

    def sort_key(r):
        # ranked students and reference entries sorted together by score; unranked students last
        group = 1 if r["unranked"] is not None and not r["reference"] else 0
        return (group, -(sign * r["value"]) if r["value"] is not None else float("inf"))

    entries.sort(key=sort_key)
    rank = 0
    for r in entries:
        if r["unranked"] is None and not r["reference"]:
            rank += 1
            r["rank"] = rank
        else:
            r["rank"] = None
    return entries


def _better(a, b, sign):
    """Ranked beats unranked; then the better value."""
    if (a["unranked"] is None) != (b["unranked"] is None):
        return a["unranked"] is None
    if a["value"] is None:
        return False
    return b["value"] is None or sign * a["value"] > sign * b["value"]
