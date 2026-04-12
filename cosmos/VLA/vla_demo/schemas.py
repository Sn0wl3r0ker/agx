from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SchemaError(ValueError):
    """Raised when a mission schema is invalid."""


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise SchemaError(f"'{key}' must be an integer")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"'{key}' must be a non-empty string")
    return value.strip()


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SchemaError(f"'{key}' must be a list of strings")
    return [item.strip() for item in value if item.strip()]


def _float_with_default(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)):
        raise SchemaError(f"'{key}' must be numeric")
    return float(value)


def _int_with_default(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise SchemaError(f"'{key}' must be an integer")
    return value


@dataclass(frozen=True)
class StepSpec:
    step_id: int
    instruction: str
    visual_goal: str
    expected_landmarks: list[str] = field(default_factory=list)
    control_primitive: str = "move_forward_until_recheck"
    votes_needed: int = 3
    confidence_threshold: float = 0.75
    min_dwell_sec: float = 2.0
    timeout_sec: float = 10.0
    fallback: str = "pause"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepSpec":
        if not isinstance(data, dict):
            raise SchemaError("Step must be a dictionary")
        step = cls(
            step_id=_require_int(data, "step_id"),
            instruction=_require_str(data, "instruction"),
            visual_goal=_require_str(data, "visual_goal"),
            expected_landmarks=_optional_str_list(data, "expected_landmarks"),
            control_primitive=_require_str(data, "control_primitive"),
            votes_needed=_int_with_default(data, "votes_needed", 3),
            confidence_threshold=_float_with_default(data, "confidence_threshold", 0.75),
            min_dwell_sec=_float_with_default(data, "min_dwell_sec", 2.0),
            timeout_sec=_float_with_default(data, "timeout_sec", 10.0),
            fallback=_require_str(data, "fallback"),
        )
        if step.votes_needed < 1:
            raise SchemaError("'votes_needed' must be >= 1")
        if not 0.0 <= step.confidence_threshold <= 1.0:
            raise SchemaError("'confidence_threshold' must be between 0 and 1")
        if step.min_dwell_sec < 0 or step.timeout_sec <= 0:
            raise SchemaError("'min_dwell_sec' must be >= 0 and 'timeout_sec' > 0")
        return step


@dataclass(frozen=True)
class MissionSpec:
    mission_id: str
    mission_text: str
    environment_id: str
    camera_source: str
    inference_interval_sec: float
    steps: list[StepSpec]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MissionSpec":
        if not isinstance(data, dict):
            raise SchemaError("Mission file must contain a dictionary at the root")
        steps_raw = data.get("steps")
        if not isinstance(steps_raw, list) or not steps_raw:
            raise SchemaError("'steps' must be a non-empty list")
        steps = [StepSpec.from_dict(item) for item in steps_raw]
        step_ids = [step.step_id for step in steps]
        if len(step_ids) != len(set(step_ids)):
            raise SchemaError("Step IDs must be unique")
        return cls(
            mission_id=_require_str(data, "mission_id"),
            mission_text=_require_str(data, "mission_text"),
            environment_id=_require_str(data, "environment_id"),
            camera_source=_require_str(data, "camera_source"),
            inference_interval_sec=_float_with_default(
                data, "inference_interval_sec", 1.5
            ),
            steps=steps,
        )

    def get_step(self, step_index: int) -> StepSpec:
        return self.steps[step_index]

