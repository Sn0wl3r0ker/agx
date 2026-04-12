from __future__ import annotations

from pathlib import Path
import json

import yaml

from .schemas import MissionSpec


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_mission_path() -> Path:
    return package_root() / "config" / "sample_mission.yaml"


def load_mission_file(path: str | Path) -> MissionSpec:
    mission_path = Path(path).expanduser().resolve()
    raw = mission_path.read_text(encoding="utf-8")
    if mission_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    return MissionSpec.from_dict(payload)
