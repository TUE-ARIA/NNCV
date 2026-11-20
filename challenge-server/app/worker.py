"""The worker: runs one submission at a time on the reserved GPU, then scores it.

    python -m app.worker
"""
import json
import logging
import shutil
import time
import traceback
from pathlib import Path

from app import db, rules, sandbox
from app.packages import PackageError, extract_zip
from app.settings import config, data_dir

log = logging.getLogger("worker")
POLL_SECONDS = 3
SCORE_TIMEOUT_S = 3600


def job_dir(sub_id: int) -> Path:
    return data_dir() / "jobs" / str(sub_id)


def eval_dirs(track: str, test_set: str) -> tuple[Path, Path]:
    """(input folder given to the submission, ground-truth folder given to the scorer)."""
    root = Path(config()["paths"]["eval_data"][test_set])
    if track == "generative":
        return root / "generative" / "masks", root / "generative"
    return root / "segmentation" / "images", root / "segmentation" / "labels"


def run_submission(sub: dict) -> dict:
    """Inference in the sandbox. Returns fields to store; raises PackageError on bad input."""
    cfg, track = config(), sub["track"]
    work = job_dir(sub["id"])
    output = work / "output"
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o777)                               # the job runs as a non-root user
    inputs, _ = eval_dirs(track, sub["test_set"])
    mounts = {**sandbox.harness_mount(), inputs: ("/data", "ro"), output: ("/output", "rw")}
    timeout = cfg["tracks"][track]["time_limit_min"] * 60
    env = {}
    tag = None

    if sub["kind"] == "zip":
        code = work / "submission"
        manifest = extract_zip(Path(sub["upload_path"]), code)
        image = cfg["base_images"][manifest["base_image"]]["image"]
        mounts[code] = ("/submission", "ro")
        env["PYTHONPATH"] = f"/submission:/tmp/pydeps:{sandbox.HARNESS}"
        entrypoint = ["sh", "-c"]
        command = [sandbox.zip_script(track, f"/submission/{manifest['checkpoint']}", wheels=(code / "wheels").is_dir())]
    else:                                             # custom image: code in /app, weights in /app/model.pt
        tag = sandbox.load_image(Path(sub["upload_path"]), f"nncv-custom:{sub['id']}")
        image = tag
        env["PYTHONPATH"] = f"/app:{sandbox.HARNESS}"
        entrypoint = ["python"]                       # replaces any entrypoint of the student's image
        command = sandbox.harness_args(track, "/app/model.pt")
    try:
        code_, text, seconds = sandbox.run(image, command, mounts, timeout, env=env, entrypoint=entrypoint)
    finally:
        if tag:
            sandbox.remove_image(tag)
    (work / "inference.log").write_text(text, encoding="utf-8")
    if code_ is None:
        return {"ok": False, "error": f"Time limit of {timeout // 60} minutes exceeded.", "seconds": seconds}
    if code_ != 0:
        return {"ok": False, "error": f"The submission stopped with exit code {code_}. See the log.", "seconds": seconds}
    return {"ok": True, "seconds": seconds}


def score(sub: dict) -> dict:
    """Run evaluate.py in the trusted scorer container; returns the scores dict."""
    cfg, track = config(), sub["track"]
    work = job_dir(sub["id"])
    out = work / "score"
    out.mkdir(exist_ok=True)
    out.chmod(0o777)
    _, gt = eval_dirs(track, sub["test_set"])
    t = cfg["tracks"][track]
    mounts = {**sandbox.harness_mount(), work / "output": ("/pred", "ro"), gt: ("/gt", "ro"), out: ("/score", "rw"),
              Path(cfg["paths"]["torch_cache"]): ("/torch-cache", "ro")}
    command = [f"{sandbox.HARNESS}/evaluate.py", track, "--pred-dir", "/pred", "--gt-dir", "/gt",
               "--out", "/score/score.json", "--min-fps", str(t.get("min_fps", 0)),
               "--mask-dice-floor", str(t.get("mask_dice_floor", 0))]
    if track == "generative":
        mounts[Path(cfg["paths"]["reference_model"])] = ("/reference.pth", "ro")
        command += ["--reference-model", "/reference.pth"]
    code, text, _ = sandbox.run(cfg["scorer_image"], command, mounts, SCORE_TIMEOUT_S, entrypoint=["python"],
                                env={"PYTHONPATH": sandbox.HARNESS, "TORCH_HOME": "/torch-cache"})
    (work / "score.log").write_text(text, encoding="utf-8")
    result = out / "score.json"
    if code != 0 or not result.exists():
        raise RuntimeError("Scoring failed. This is usually caused by missing or malformed predictions; see the log.")
    return json.loads(result.read_text())


def process(sub: dict):
    log.info("job %s: %s %s by %s", sub["id"], sub["track"], sub["test_set"], sub["student"])
    db.update_submission(sub["id"], status="running", started_at=db.stamp(), error=None)
    fields = {}
    try:
        result = run_submission(sub)
        fields["wall_seconds"] = round(result["seconds"], 1)
        if not result["ok"]:
            fields.update(status="failed", error=result["error"])
        else:
            scores = score(sub)
            scores["wall_seconds"] = fields["wall_seconds"]
            fields.update(status="succeeded", scores=scores)
    except PackageError as e:
        fields.update(status="failed", error=str(e))
    except Exception as e:                            # keep the worker alive, record the reason
        log.error("job %s crashed:\n%s", sub["id"], traceback.format_exc())
        fields.update(status="failed", error=str(e)[:500] or "Internal error while running the submission.")
    fields["finished_at"] = db.stamp()
    db.update_submission(sub["id"], **fields)
    cleanup(sub)
    log.info("job %s: %s", sub["id"], fields["status"])


def cleanup(sub: dict):
    paths = config()["paths"]
    work = job_dir(sub["id"])
    shutil.rmtree(work / "submission", ignore_errors=True)
    if not paths.get("keep_outputs"):
        shutil.rmtree(work / "output", ignore_errors=True)
    if sub["kind"] == "image" or not paths.get("keep_uploads", True):
        Path(sub["upload_path"]).unlink(missing_ok=True)


def recover():
    """Jobs that were running when the worker stopped go back into the queue."""
    for sub in db.submissions("s.status = 'running'"):
        db.update_submission(sub["id"], status="queued", started_at=None)
        log.warning("job %s re-queued after restart", sub["id"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db.init()
    recover()
    log.info("worker started")
    while True:
        job = rules.next_job()
        if job:
            process(job)
        else:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
