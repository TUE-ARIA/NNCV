"""Reads config.yaml. The file is re-read when it changes, so most settings apply without a restart."""
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

CONFIG_PATH = Path(os.environ.get("NNCV_CONFIG", Path(__file__).parent.parent / "config.yaml"))
TEST_SETS = ("validation", "hidden")

_cache = {"mtime": None, "data": None}


def config() -> dict:
    """The current config.yaml as a dict (cached until the file changes)."""
    mtime = CONFIG_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["data"] = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        _cache["mtime"] = mtime
    return _cache["data"]


def tz() -> ZoneInfo:
    return ZoneInfo(config()["site"]["timezone"])


def now() -> datetime:
    return datetime.now(tz())


def week_start(moment: datetime | None = None) -> datetime:
    """Monday 00:00 of the week that contains `moment` (default: now)."""
    moment = moment or now()
    monday = moment - timedelta(days=moment.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time(value) -> datetime | None:
    """A config date ("2027-04-02 23:59") as a timezone-aware datetime."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=tz())
    return datetime.strptime(str(value), "%Y-%m-%d %H:%M").replace(tzinfo=tz())


def data_dir() -> Path:
    return Path(config()["paths"]["data_dir"])


def tracks() -> dict:
    return config()["tracks"]
