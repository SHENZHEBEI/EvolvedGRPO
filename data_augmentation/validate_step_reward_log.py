"""Validate the durable per-training-step reward JSONL artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _finite_number(value: Any, field: str, line_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"line {line_number}: {field} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"line {line_number}: {field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"line {line_number}: {field} is not finite")
    return result


def validate_step_reward_log(
    path: Path,
    expected_steps: int,
    expected_model_path: str | None = None,
) -> None:
    if expected_steps <= 0:
        raise ValueError("expected_steps must be positive")
    path = path.expanduser().resolve(strict=True)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"line {line_number}: blank JSONL record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(record, dict) or record.get("record_type") != "train_step":
            raise ValueError(f"line {line_number}: not a train_step record")
        records.append(record)

    if len(records) != expected_steps:
        raise ValueError(
            f"expected {expected_steps} step records, found {len(records)} in {path}"
        )
    actual_steps = [record.get("global_step") for record in records]
    wanted_steps = list(range(1, expected_steps + 1))
    if actual_steps != wanted_steps:
        raise ValueError(f"global steps are {actual_steps}, expected {wanted_steps}")

    for line_number, record in enumerate(records, 1):
        run = record.get("run")
        reward = record.get("reward")
        if not isinstance(run, Mapping) or not isinstance(reward, Mapping):
            raise ValueError(f"line {line_number}: missing run/reward object")
        if expected_model_path is not None and run.get("model_path") != expected_model_path:
            raise ValueError(
                f"line {line_number}: model_path {run.get('model_path')!r} "
                f"does not match {expected_model_path!r}"
            )
        raw_min = _finite_number(reward.get("raw_score_min"), "raw_score_min", line_number)
        raw_mean = _finite_number(reward.get("raw_score_mean"), "raw_score_mean", line_number)
        raw_max = _finite_number(reward.get("raw_score_max"), "raw_score_max", line_number)
        post_min = _finite_number(reward.get("post_kl_min"), "post_kl_min", line_number)
        post_mean = _finite_number(reward.get("post_kl_mean"), "post_kl_mean", line_number)
        post_max = _finite_number(reward.get("post_kl_max"), "post_kl_max", line_number)
        if not raw_min <= raw_mean <= raw_max:
            raise ValueError(f"line {line_number}: invalid raw reward min/mean/max ordering")
        if not post_min <= post_mean <= post_max:
            raise ValueError(f"line {line_number}: invalid post-KL reward min/mean/max ordering")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--expected-model-path")
    args = parser.parse_args()
    try:
        validate_step_reward_log(
            args.path,
            args.expected_steps,
            args.expected_model_path,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Step reward log validation failed: {exc}") from exc
    print(f"Step reward log validation OK: {args.path} ({args.expected_steps} steps)")


if __name__ == "__main__":
    main()
