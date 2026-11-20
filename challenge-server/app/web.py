"""The challenge website: leaderboards, uploads, queue and personal results.

    uvicorn app.web:app --host 0.0.0.0 --port 8000
"""
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app import db, rules
from app.packages import PackageError, check_image_tar, check_zip
from app.security import check_password, hash_password, new_password
from app.settings import TEST_SETS, config, data_dir, now, parse_time, week_start

HERE = Path(__file__).parent
app = FastAPI(title="NNCV challenge", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("NNCV_SECRET", "change-me"), max_age=8 * 3600)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


@app.on_event("startup")
def startup():
    db.init()
    (data_dir() / "tmp").mkdir(parents=True, exist_ok=True)      # TMPDIR for large uploads
    if os.environ.get("NNCV_SECRET", "change-me") in ("", "change-me"):
        print("WARNING: NNCV_SECRET is not set in .env; login cookies are not secure.")


# ------------------------------------------------------------------ template helpers
def fmt(metric, value):
    if value is None:
        return "–"
    if metric in ("fps", "fvd", "fid", "wall_seconds"):
        return f"{value:.1f}"
    return f"{value:.3f}"


METRIC_LABELS = {"dice": "Dice", "consistency_score": "Consistency score", "change_dice": "Change Dice", "fps": "FPS",
                 "fvd": "FVD", "fid": "FID", "mask_dice": "Mask Dice", "mask_dice_real": "Real frames",
                 "wall_seconds": "Run time (s)"}
templates.env.filters["fmt"] = lambda v, m: fmt(m, v)
templates.env.globals.update(METRIC_LABELS=METRIC_LABELS)


def signed_in(request) -> dict | None:
    """The signed-in account, if any."""
    sid = request.session.get("student_id")
    return db.student_by_id(sid) if sid else None


def is_staff(request) -> bool:
    me = signed_in(request)
    return bool(me and me["is_staff"] and me["active"])


def render(request, name, **ctx):
    cfg = config()
    me = signed_in(request)
    ctx.update(cfg=cfg, tracks=cfg["tracks"], me=me, staff=bool(me and me["is_staff"] and me["active"]),
               deadline=parse_time(cfg["submissions"].get("deadline")))
    return templates.TemplateResponse(request, name, ctx)


def message(request, title, text, status=400):
    resp = render(request, "message.html", title=title, text=text)
    resp.status_code = status
    return resp


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    tops = {t: rules.leaderboard(t, "validation")[:3] for t in config()["tracks"]}
    return render(request, "home.html", tops=tops, queue=rules.queue())


@app.get("/track/{track}", response_class=HTMLResponse)
def track_page(request: Request, track: str, set: str = "validation", student: str = ""):
    if track not in config()["tracks"] or set not in TEST_SETS:
        return message(request, "Not found", "This track or test set does not exist.", 404)
    return render(request, "track.html", track=track, test_set=set, student=student,
                  students=db.students(), rows=rules.leaderboard(track, set, student or None),
                  show_scores=scores_visible(request, set))


@app.get("/partials/leaderboard/{track}", response_class=HTMLResponse)
def leaderboard_partial(request: Request, track: str, set: str = "validation", student: str = ""):
    if track not in config()["tracks"] or set not in TEST_SETS:
        return HTMLResponse("", status_code=404)
    return render(request, "_leaderboard.html", track=track, test_set=set, student=student,
                  rows=rules.leaderboard(track, set, student or None), show_scores=scores_visible(request, set))


@app.get("/queue", response_class=HTMLResponse)
def queue_page(request: Request):
    return render(request, "queue.html", queue=rules.queue())


@app.get("/partials/queue", response_class=HTMLResponse)
def queue_partial(request: Request):
    return render(request, "_queue.html", queue=rules.queue())


@app.get("/help", response_class=HTMLResponse)
def help_page(request: Request):
    return render(request, "help.html")


# ------------------------------------------------------------------ submitting
def save_upload(upload: UploadFile, dest: Path, limit_bytes: float) -> int:
    """Copy the upload in chunks; stop as soon as it exceeds the limit."""
    size = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        while chunk := upload.file.read(16 * 1024 * 1024):
            size += len(chunk)
            if size > limit_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise PackageError(f"The file is larger than the limit of {limit_bytes / 1e9:.0f} GB.")
            f.write(chunk)
    return size


@app.post("/submit/{track}")
def submit(request: Request, track: str, student: str = Form(...), password: str = Form(...),
           test_set: str = Form("validation"), kind: str = Form("zip"), note: str = Form(""),
           file: UploadFile = File(...)):
    cfg = config()
    st = db.student_by_name(student)
    if not st or not check_password(password, st["password_hash"]):
        time.sleep(1)                                     # slows down password guessing
        return message(request, "Not submitted", "The name or password is not correct.", 403)
    if kind not in ("zip", "image") or (kind == "image" and not cfg["submissions"]["allow_custom_images"]):
        return message(request, "Not submitted", "This kind of submission is not allowed.")
    reason = rules.submission_check(st, track, test_set)
    if reason:
        return message(request, "Not submitted", reason)

    limit = cfg["submissions"]["max_upload_gb" if kind == "zip" else "max_custom_image_gb"] * 1e9
    stamp = now().strftime("%Y%m%d-%H%M%S")
    dest = data_dir() / "uploads" / st["name"].replace("/", "_") / f"{track}-{test_set}-{stamp}.{'zip' if kind == 'zip' else 'tar'}"
    try:
        size = save_upload(file, dest, limit)
        manifest = check_zip(dest, track) if kind == "zip" else (check_image_tar(dest) or {})
    except PackageError as e:
        dest.unlink(missing_ok=True)
        return message(request, "Not submitted", str(e))

    reason = rules.submission_check(st, track, test_set)  # again: another upload may have finished meanwhile
    if reason:
        dest.unlink(missing_ok=True)
        return message(request, "Not submitted", reason)
    sub_id = db.add_submission(st["id"], track, test_set, kind, manifest.get("base_image"), note.strip()[:200], dest, size)
    request.session["student_id"] = st["id"]              # signed in, so the log is visible on the result page
    return RedirectResponse(f"/submission/{sub_id}", status_code=303)


# ------------------------------------------------------------------ results
def scores_visible(request, test_set: str) -> bool:
    """Scores of a test set are shown once revealed in config.yaml, and always to staff."""
    return config()["test_sets"][test_set]["show_scores"] or is_staff(request)


@app.get("/submission/{sub_id}", response_class=HTMLResponse)
def submission_page(request: Request, sub_id: int):
    sub = db.submission(sub_id)
    if not sub:
        return message(request, "Not found", "This submission does not exist.", 404)
    owner = request.session.get("student_id") == sub["student_id"]
    queue = rules.queue()
    position = next((i for i, q in enumerate(queue) if q["id"] == sub_id), None)
    return render(request, "submission.html", sub=sub, owner=owner or is_staff(request), position=position,
                  show_scores=scores_visible(request, sub["test_set"]))


@app.get("/submission/{sub_id}/log", response_class=PlainTextResponse)
def submission_log(request: Request, sub_id: int):
    sub = db.submission(sub_id)
    staff = is_staff(request)
    if not sub or (request.session.get("student_id") != sub["student_id"] and not staff):
        return PlainTextResponse("Sign in on 'My submissions' to see your logs.", status_code=403)
    work = data_dir() / "jobs" / str(sub_id)
    # the scorer log is for staff only: it can reveal ground-truth paths
    parts = []
    for name in ("inference.log", "score.log") if staff else ("inference.log",):
        f = work / name
        if f.exists():
            text = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"===== {name}\n{text}" if staff else text)
    return PlainTextResponse("\n".join(parts) or "No log yet.")


@app.get("/me", response_class=HTMLResponse)
def me_page(request: Request):
    sid = request.session.get("student_id")
    if not sid:
        return render(request, "login.html", students=db.students())
    subs = db.submissions("s.student_id = ?", (sid,))
    quota = {t: {s: rules.quota_used(sid, t, s) for s in TEST_SETS} for t in config()["tracks"]}
    return render(request, "me.html", subs=subs, quota=quota, week=week_start())


@app.post("/me")
def login(request: Request, student: str = Form(...), password: str = Form(...)):
    st = db.student_by_name(student)
    if not st or not check_password(password, st["password_hash"]):
        time.sleep(1)
        return message(request, "Not signed in", "The name or password is not correct.", 403)
    request.session["student_id"] = st["id"]
    return RedirectResponse("/me", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# ------------------------------------------------------------------ staff
def back_to_admin(request, text):
    """Back to the admin page with the same filters, showing `text` once."""
    request.session["flash"] = text
    ref = urlparse(request.headers.get("referer", ""))
    return RedirectResponse(f"/admin?{ref.query}" if ref.path == "/admin" else "/admin", status_code=303)


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, status: str = "", track: str = "", student: str = ""):
    if not is_staff(request):
        return message(request, "Staff only", "Sign in with a staff account to see this page.", 403)
    where, params = ["1=1"], []
    for column, value in (("s.status", status), ("s.track", track), ("st.name", student)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)
    subs = db.submissions(" AND ".join(where), tuple(params), limit=300)
    return render(request, "admin.html", page="admin", subs=subs, students=db.students(include_inactive=True),
                  status=status, track_filter=track, student=student, flash=request.session.pop("flash", None))


@app.post("/admin/submission/{sub_id}/{action}")
def admin_submission(request: Request, sub_id: int, action: str):
    if not is_staff(request):
        return message(request, "Staff only", "Sign in with a staff account to do this.", 403)
    sub = db.submission(sub_id)
    if not sub or action not in ("cancel", "requeue"):
        return message(request, "Not found", "This submission or action does not exist.", 404)
    reason = rules.cancel(sub) if action == "cancel" else rules.requeue(sub)
    return back_to_admin(request, reason or f"Submission {sub_id}: {'cancelled' if action == 'cancel' else 'queued again'}.")


@app.post("/admin/student/{sid}/{action}")
def admin_student(request: Request, sid: int, action: str):
    if not is_staff(request):
        return message(request, "Staff only", "Sign in with a staff account to do this.", 403)
    st = db.student_by_id(sid)
    if not st or action not in ("password", "activate", "deactivate"):
        return message(request, "Not found", "This account or action does not exist.", 404)
    if action == "password":
        pw = new_password()
        db.update_student(sid, password_hash=hash_password(pw))
        return render(request, "message.html", title="New password",
                      text=f"New password for {st['name']}: {pw}  (shown once; the old password no longer works)")
    if sid == request.session.get("student_id"):
        return back_to_admin(request, "You cannot deactivate your own account.")
    db.update_student(sid, active=int(action == "activate"))
    return back_to_admin(request, f"{st['name']}: {action}d.")
