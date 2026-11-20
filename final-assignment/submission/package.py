"""Builds the zip you upload to the challenge server.

Run from the `final-assignment` folder:

    python -m submission.package --track consistency --checkpoint checkpoints/my-run/best.pt
    python -m submission.package --track generative --checkpoint checkpoints/gen/last.pt --base-image generative

The zip contains your track folder, every local package it imports (for example `segmentation/`
when the consistency model reuses its U-Net), your weights as `model.pt` (or a folder of weights
as `checkpoint/`), optional wheels, and a `submission.yaml` that tells the server what to run.
"""
import re
import zipfile
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the final-assignment folder
LOCAL_PACKAGES = ("segmentation", "consistency", "realtime", "generative", "atlas")
SKIP = {"__pycache__", ".ipynb_checkpoints"}
IMPORT = re.compile(r"^\s*(?:from|import)\s+(\w+)", re.MULTILINE)


def local_imports(package: str) -> set[str]:
    """Local packages imported by the .py files of `package`, followed recursively."""
    found, todo = {package}, [package]
    while todo:
        for py in (ROOT / todo.pop()).rglob("*.py"):
            for name in IMPORT.findall(py.read_text(encoding="utf-8", errors="ignore")):
                if name in LOCAL_PACKAGES and name not in found:
                    found.add(name)
                    todo.append(name)
    return found


def add_folder(zf: zipfile.ZipFile, folder: Path, arc_root: str):
    for f in sorted(folder.rglob("*")):
        if f.is_file() and not SKIP.intersection(f.parts) and f.suffix != ".pyc":
            zf.write(f, f"{arc_root}/{f.relative_to(folder).as_posix()}")


def main():
    p = ArgumentParser(description="Build a challenge submission zip")
    p.add_argument("--track", required=True, choices=["segmentation", "consistency", "realtime", "generative"])
    p.add_argument("--checkpoint", required=True, type=Path, help="Weights file, or a folder of weights")
    p.add_argument("--base-image", default="pytorch", help="pytorch, transformers or generative (see the server's help page)")
    p.add_argument("--wheels", type=Path, help="Folder with extra .whl files, installed offline on the server")
    p.add_argument("--out", type=Path, help="Output zip (default: submission_<track>.zip)")
    args = p.parse_args()

    out = args.out or Path(f"submission_{args.track}.zip")
    packages = sorted(local_imports(args.track))
    checkpoint_name = "checkpoint" if args.checkpoint.is_dir() else "model.pt"
    manifest = f"track: {args.track}\nbase_image: {args.base_image}\ncheckpoint: {checkpoint_name}\n"

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.yaml", manifest)
        for pkg in packages:
            add_folder(zf, ROOT / pkg, pkg)
        if args.checkpoint.is_dir():
            add_folder(zf, args.checkpoint, checkpoint_name)
        else:
            zf.write(args.checkpoint, checkpoint_name)
        if args.wheels:
            for whl in sorted(args.wheels.glob("*.whl")):
                zf.write(whl, f"wheels/{whl.name}")

    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {', '.join(packages)}, {checkpoint_name}"
          + (", wheels" if args.wheels else ""))


if __name__ == "__main__":
    main()
