"""Read and validate the public ARC episode-bundle contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arc-episode-bundle-v1"
FORBIDDEN_KEYS = {
    "observations",
    "central_state",
    "model_parameters",
    "training_configuration",
    "checkpoint_path",
}


class BundleValidationError(ValueError):
    """Raised when a file does not satisfy the public bundle contract."""


def _require(mapping: dict, keys: set[str], location: str) -> None:
    missing = keys - set(mapping)
    if missing:
        raise BundleValidationError(f"Missing {sorted(missing)} at {location}")


def _check_forbidden(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                raise BundleValidationError(f"Forbidden key {key!r} at {location}")
            _check_forbidden(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_forbidden(item, f"{location}[{index}]")


def validate_bundle(bundle: Any) -> dict:
    if not isinstance(bundle, dict):
        raise BundleValidationError("Bundle root must be an object")
    _require(bundle, {"schema_version", "generator", "episode"}, "root")
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise BundleValidationError(
            f"Unsupported schema {bundle['schema_version']!r}; expected {SCHEMA_VERSION!r}"
        )
    episode = bundle["episode"]
    if not isinstance(episode, dict):
        raise BundleValidationError("episode must be an object")
    _require(episode, {"id", "scenario", "policy", "map", "timeline", "metrics", "disclosure"}, "episode")
    _require(episode["scenario"], {"id", "category", "bank_id", "split", "world_seed", "horizon_h", "time_step_h"}, "episode.scenario")
    _require(episode["policy"], {"id", "label", "kind", "execution_information", "checkpoint_id"}, "episode.policy")
    if not isinstance(episode["metrics"], dict):
        raise BundleValidationError("episode.metrics must be an object")
    timeline = episode["timeline"]
    if not isinstance(timeline, list) or not timeline:
        raise BundleValidationError("episode.timeline must be a non-empty list")
    previous_step = -1
    previous_time = float("-inf")
    for index, frame in enumerate(timeline):
        if not isinstance(frame, dict):
            raise BundleValidationError(f"timeline[{index}] must be an object")
        _require(
            frame,
            {"step", "time_h", "reward", "cumulative_reward", "reward_components", "vessels", "incidents", "decisions", "events"},
            f"timeline[{index}]",
        )
        if frame["step"] <= previous_step or frame["time_h"] < previous_time:
            raise BundleValidationError("Timeline steps and times must increase")
        for list_key in ("vessels", "incidents", "decisions", "events"):
            if not isinstance(frame[list_key], list):
                raise BundleValidationError(f"timeline[{index}].{list_key} must be a list")
        previous_step = frame["step"]
        previous_time = frame["time_h"]
    disclosure = episode["disclosure"]
    for flag in (
        "contains_policy_observations",
        "contains_central_critic_state",
        "contains_model_parameters",
        "contains_training_configuration",
    ):
        if disclosure.get(flag) is not False:
            raise BundleValidationError(f"Disclosure flag {flag!r} must be false")
    _check_forbidden(bundle)
    return bundle


def load_bundle(path: str | Path) -> dict:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"Could not read {source.name}: {error}") from error
    return validate_bundle(payload)


def load_bundle_bytes(data: bytes, name: str = "uploaded file") -> dict:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"Could not read {name}: {error}") from error
    return validate_bundle(payload)

