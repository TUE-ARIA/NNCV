"""Runs submissions and the scorer in locked-down Docker containers on the reserved GPU.

Student code never gets network access, sees only the input data (never the ground truth),
and runs the fixed harness from this repository, mounted read-only.
"""
import shlex
import time
from pathlib import Path

import docker
from docker.types import DeviceRequest

from app.settings import config

HARNESS = "/harness"           # final-assignment/ of the repository, read-only
USER = "1000:1000"
LABEL = "nncv.job"             # marks job and scorer containers, to find leftovers after a restart


def client():
    return docker.from_env(timeout=600)


def _limits() -> dict:
    sb = config()["sandbox"]
    return {
        "device_requests": [DeviceRequest(device_ids=[str(sb["gpu"])], capabilities=[["gpu"]])],
        "network_mode": "none",
        "read_only": True,
        "user": USER,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "mem_limit": f"{sb['memory_gb']}g",
        "memswap_limit": f"{sb['memory_gb']}g",
        "shm_size": f"{sb['shm_gb']}g",
        "nano_cpus": int(sb["cpus"] * 1e9),
        "pids_limit": sb["pids"],
        "tmpfs": {"/tmp": f"size={sb['tmp_gb']}g,mode=1777"},
        "environment": {"HOME": "/tmp", "TORCH_HOME": "/tmp/torch", "XDG_CACHE_HOME": "/tmp/cache",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    }


def run(image: str, command: list[str], mounts: dict, timeout_s: float, env: dict | None = None,
        entrypoint=None) -> tuple[int | None, str, float]:
    """Run a container to completion. Returns (exit code or None on timeout, log, seconds)."""
    opts = _limits()
    opts["environment"].update(env or {})
    volumes = {str(host): {"bind": inside, "mode": mode} for host, (inside, mode) in mounts.items()}
    c = client().containers.run(image, command, volumes=volumes, detach=True, entrypoint=entrypoint,
                                labels={LABEL: "1"}, **opts)
    start = time.monotonic()
    try:
        result = c.wait(timeout=timeout_s)
        code = result.get("StatusCode")
    except Exception:                                  # timeout (requests ReadTimeout) or daemon error
        code = None
        c.kill()
    seconds = time.monotonic() - start
    log = c.logs(stdout=True, stderr=True).decode(errors="replace")
    c.remove(force=True)
    return code, _tail(log), seconds


def remove_leftovers() -> int:
    """Remove job containers left running when the worker stopped; returns how many."""
    leftovers = client().containers.list(all=True, filters={"label": LABEL})
    for c in leftovers:
        c.remove(force=True)
    return len(leftovers)


def _tail(log: str) -> str:
    limit = config()["sandbox"]["max_log_kb"] * 1024
    return log if len(log) <= limit else "[... start of log removed ...]\n" + log[-limit:]


def harness_args(track: str, checkpoint: str) -> list[str]:
    """Arguments of the fixed harness, the same for every kind of submission."""
    return [f"{HARNESS}/submission/run.py", "--track", track, "--input-dir", "/data", "--output-dir", "/output",
            "--checkpoint", checkpoint]


def zip_script(track: str, checkpoint: str, wheels: bool) -> str:
    """Shell script for a zip submission: optional offline wheels, then the harness."""
    install = ("pip install --quiet --no-index --no-deps --target /tmp/pydeps /submission/wheels/*.whl && "
               if wheels else "")
    return install + "exec python " + shlex.join(harness_args(track, checkpoint))   # quoted: names come from the student


def load_image(tar_path: Path, tag: str) -> str:
    """Load a student's `docker save` archive and give it a unique tag."""
    with open(tar_path, "rb") as f:                    # streamed, not read into memory
        image = client().images.load(f)[0]
    image.tag(tag)
    return tag


def remove_image(tag: str):
    try:
        client().images.remove(tag, force=True)
    except docker.errors.ImageNotFound:
        pass


def harness_mount() -> dict:
    return {Path(config()["paths"]["harness_dir"]): (HARNESS, "ro")}
