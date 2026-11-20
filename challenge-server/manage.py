"""Staff commands for the challenge server. Run inside the worker container (it has Docker access):

    docker compose exec worker python manage.py <command> [options]

Commands:
    import-students roster.csv passwords.csv   add students (columns: name,email); writes their new passwords
    reset-password "Name"                      new password for one student
    add-baseline "Baseline"                    staff entry, shown on leaderboards but never ranked
    submit "Name" TRACK FILE [--set hidden]    queue a submission for anyone, without quota (e.g. baselines)
    list [--status queued]                     recent submissions
    cancel ID                                  cancel a waiting submission
    requeue ID                                 run a submission again
    pull-images                                download the base and scorer images
    prefetch-weights                           download the FID/FVD feature networks into the torch cache
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

from app import db
from app.packages import check_image_tar, check_zip
from app.security import hash_password, new_password
from app.settings import config, data_dir


def import_students(args):
    rows = list(csv.DictReader(open(args.roster, newline="", encoding="utf-8-sig")))
    if not rows or "name" not in rows[0]:
        sys.exit("The roster needs a header row with at least the column 'name' (and optionally 'email').")
    out = []
    for r in rows:
        name = r["name"].strip()
        if not name:
            continue
        pw = new_password()
        db.upsert_student(name, (r.get("email") or "").strip(), hash_password(pw))
        out.append({"name": name, "email": r.get("email", ""), "password": pw})
    with open(args.passwords, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "email", "password"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} students imported. Passwords written to {args.passwords}: send them, then delete the file.")


def reset_password(args):
    st = db.student_by_name(args.name) or sys.exit(f"No student named {args.name!r}.")
    pw = new_password()
    db.upsert_student(st["name"], st["email"], hash_password(pw), bool(st["is_baseline"]))
    print(f"New password for {st['name']}: {pw}")


def add_baseline(args):
    db.upsert_student(args.name, "", hash_password(new_password()), is_baseline=True)
    print(f"Baseline entry {args.name!r} added. Submit for it with: manage.py submit {args.name!r} TRACK FILE")


def submit(args):
    st = db.student_by_name(args.name) or sys.exit(f"No student named {args.name!r}.")
    src = Path(args.file)
    kind = "image" if src.suffix == ".tar" else "zip"
    manifest = check_zip(src, args.track) if kind == "zip" else (check_image_tar(src) or {})
    dest = data_dir() / "uploads" / st["name"] / f"{args.track}-{args.set}-staff-{src.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    sid = db.add_submission(st["id"], args.track, args.set, kind, manifest.get("base_image"), args.note, dest, dest.stat().st_size)
    print(f"Submission {sid} queued.")


def list_subs(args):
    where, params = ("s.status = ?", (args.status,)) if args.status else ("1=1", ())
    for s in db.submissions(where, params, limit=args.n):
        print(f"{s['id']:>5}  {s['created_at'][:16]}  {s['status']:<9}  {s['track']:<12} {s['test_set']:<10} {s['student']}"
              + (f"  ({s['error'][:60]})" if s["error"] else ""))


def cancel(args):
    s = db.submission(args.id) or sys.exit("No such submission.")
    if s["status"] != "queued":
        sys.exit(f"Submission {args.id} is {s['status']}; only waiting submissions can be cancelled.")
    db.update_submission(args.id, status="cancelled", finished_at=db.stamp(), error="Cancelled by staff.")
    print("Cancelled.")


def requeue(args):
    s = db.submission(args.id) or sys.exit("No such submission.")
    if not Path(s["upload_path"]).exists():
        sys.exit("The upload of this submission no longer exists.")
    db.update_submission(args.id, status="queued", started_at=None, finished_at=None, scores=None, error=None)
    print("Queued again.")


def pull_images(args):
    import docker
    cli = docker.from_env()
    images = {b["image"] for b in config()["base_images"].values()} | {config()["scorer_image"]}
    for image in sorted(images):
        print("pulling", image)
        cli.images.pull(image)


def prefetch_weights(args):
    """Download the torchvision weights used by FID and FVD; the scorer itself has no internet."""
    import docker
    cache = Path(config()["paths"]["torch_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    code = ("from torchvision.models import inception_v3, Inception_V3_Weights; "
            "from torchvision.models.video import r3d_18, R3D_18_Weights; "
            "inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1); r3d_18(weights=R3D_18_Weights.KINETICS400_V1); "
            "print('weights cached')")
    out = docker.from_env().containers.run(config()["scorer_image"], ["python", "-c", code], entrypoint=[],
                                           environment={"TORCH_HOME": "/torch-cache"},
                                           volumes={str(cache): {"bind": "/torch-cache", "mode": "rw"}}, remove=True)
    print(out.decode())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("import-students"); c.add_argument("roster"); c.add_argument("passwords"); c.set_defaults(f=import_students)
    c = sub.add_parser("reset-password"); c.add_argument("name"); c.set_defaults(f=reset_password)
    c = sub.add_parser("add-baseline"); c.add_argument("name"); c.set_defaults(f=add_baseline)
    c = sub.add_parser("submit"); c.add_argument("name"); c.add_argument("track"); c.add_argument("file")
    c.add_argument("--set", default="validation", choices=["validation", "hidden"]); c.add_argument("--note", default="")
    c.set_defaults(f=submit)
    c = sub.add_parser("list"); c.add_argument("--status"); c.add_argument("-n", type=int, default=30); c.set_defaults(f=list_subs)
    c = sub.add_parser("cancel"); c.add_argument("id", type=int); c.set_defaults(f=cancel)
    c = sub.add_parser("requeue"); c.add_argument("id", type=int); c.set_defaults(f=requeue)
    c = sub.add_parser("pull-images"); c.set_defaults(f=pull_images)
    c = sub.add_parser("prefetch-weights"); c.set_defaults(f=prefetch_weights)
    args = p.parse_args()
    db.init()
    args.f(args)


if __name__ == "__main__":
    main()
