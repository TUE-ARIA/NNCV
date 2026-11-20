"""Checks and unpacks uploads: a zip with code and weights, or a complete Docker image (.tar).

A zip submission looks like this (made by `python -m submission.package` in the course repo):

    submission.yaml          base_image: pytorch      checkpoint: model.pt
    model.pt                 the weights
    consistency/             the track folder: predictor.py, model.py, ...
    wheels/                  optional: extra packages as .whl files, installed offline
"""
import json
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import yaml

from app.settings import config

MAX_UNPACKED_FACTOR = 3        # unpacked zip may be at most 3x the upload limit (zip bombs)


class PackageError(ValueError):
    """A problem with an upload, explained for the student."""


def _safe_name(name: str) -> bool:
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and "\\" not in name


def read_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        manifest = yaml.safe_load(zf.read("submission.yaml")) or {}
    except KeyError:
        raise PackageError("The zip has no submission.yaml at its top level. Make it with `python -m submission.package`.")
    except yaml.YAMLError as e:
        raise PackageError(f"submission.yaml is not valid YAML: {e}")
    if not isinstance(manifest, dict):
        raise PackageError("submission.yaml must contain keys such as base_image and checkpoint.")
    return manifest


def check_zip(path: Path, track: str) -> dict:
    """Validate a zip upload; returns its manifest."""
    cfg = config()
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise PackageError("The file is not a valid zip archive.")
    with zf:
        names = zf.namelist()
        bad = [n for n in names if not _safe_name(n)]
        if bad:
            raise PackageError(f"Unsafe path in zip: {bad[0]}")
        links = [i.filename for i in zf.infolist() if stat.S_ISLNK(i.external_attr >> 16)]
        if links:
            raise PackageError(f"Symbolic links are not allowed: {links[0]}")
        unpacked = sum(i.file_size for i in zf.infolist())
        if unpacked > cfg["submissions"]["max_upload_gb"] * MAX_UNPACKED_FACTOR * 1e9:
            raise PackageError("The zip unpacks to an unusually large size.")

        manifest = read_manifest(zf)
        image = manifest.get("base_image")
        if image not in cfg["base_images"]:
            raise PackageError(f"base_image must be one of: {', '.join(cfg['base_images'])} (got {image!r}).")
        if manifest.get("track") and manifest["track"] != track:
            raise PackageError(f"This zip was made for the {manifest['track']} track, not {track}.")
        if f"{track}/predictor.py" not in names:
            raise PackageError(f"The zip must contain {track}/predictor.py.")
        checkpoint = manifest.get("checkpoint", "model.pt")
        is_folder = any(n.startswith(checkpoint.rstrip("/") + "/") for n in names)   # e.g. a Hugging Face model folder
        if not _safe_name(checkpoint) or (checkpoint not in names and not is_folder):
            raise PackageError(f"The checkpoint {checkpoint!r} named in submission.yaml is not in the zip.")
        manifest.setdefault("checkpoint", checkpoint)
        return manifest


def check_image_tar(path: Path) -> None:
    """A `docker save` archive has a manifest.json at its top level."""
    try:
        with tarfile.open(path) as tf:
            member = tf.getmember("manifest.json")
            json.load(tf.extractfile(member))
    except (tarfile.TarError, KeyError, json.JSONDecodeError):
        raise PackageError("The file is not a Docker image archive. Create one with `docker save -o image.tar <name>`.")


def extract_zip(path: Path, dest: Path) -> dict:
    """Unpack a checked zip into dest; returns the manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        manifest = read_manifest(zf)
        for info in zf.infolist():
            if not _safe_name(info.filename) or stat.S_ISLNK(info.external_attr >> 16):
                raise PackageError(f"Unsafe entry in zip: {info.filename}")
            zf.extract(info, dest)
    manifest.setdefault("checkpoint", "model.pt")
    return manifest
