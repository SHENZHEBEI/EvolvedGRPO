"""Validate that a local Hugging Face model checkpoint is complete.

The coevolution launchers use this after every FSDP merge and before a model
is handed to the next training or inference stage.  A config-only directory is
not a usable checkpoint: either one non-sharded weight file or every file
listed by a sharded weight index must exist and be non-empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def validate_checkpoint(model_dir: Path, expected_model_type: str | None = None) -> None:
    model_dir = model_dir.expanduser().resolve(strict=True)
    config_path = model_dir / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise ValueError(f"missing or empty config.json: {model_dir}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid config.json: {config_path}: {exc}") from exc
    if expected_model_type and config.get("model_type") != expected_model_type:
        raise ValueError(
            f"unexpected model_type in {config_path}: "
            f"{config.get('model_type')!r}, expected {expected_model_type!r}"
        )

    for name in SINGLE_WEIGHT_FILES:
        weight = model_dir / name
        if weight.is_file() and weight.stat().st_size > 0:
            return

    index_path = next(
        (
            model_dir / name
            for name in INDEX_FILES
            if (model_dir / name).is_file() and (model_dir / name).stat().st_size > 0
        ),
        None,
    )
    if index_path is None:
        raise ValueError(f"no model weights or sharded weight index found: {model_dir}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid weight index: {index_path}: {exc}") from exc
    weight_files: Iterable[str] = set(index.get("weight_map", {}).values())
    if not weight_files:
        raise ValueError(f"empty weight_map: {index_path}")
    missing = [
        name
        for name in sorted(weight_files)
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"missing or empty weight shard(s): {preview}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--expected-model-type")
    args = parser.parse_args()
    try:
        validate_checkpoint(args.model_dir, args.expected_model_type)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Checkpoint validation failed: {exc}") from exc
    print(f"Checkpoint validation OK: {args.model_dir}")


if __name__ == "__main__":
    main()
