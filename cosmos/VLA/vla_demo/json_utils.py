import json
from typing import Any


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def loads_json(raw: str, context: str = "payload") -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {context}: {exc}") from exc
