"""Validate the fixed-size multimodal JSON dataset contract before GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REQUIRED_KEYS = ("id", "problem", "answer", "image")


def validate_multimodal_dataset(
    dataset_dir: Path,
    *,
    expected_rows: int | None = None,
    require_unique_images: bool = True,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.expanduser().resolve(strict=True)
    data_path = dataset_dir / "data.json"
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise ValueError(f"missing or empty data.json: {data_path}")
    raw = data_path.read_bytes()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {data_path}: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{data_path} must contain a non-empty JSON list")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"row count mismatch in {data_path}: {len(rows)}, expected {expected_rows}"
        )

    ids: set[str] = set()
    images: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} is not a JSON object")
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"row {index} is missing required keys: {missing}")
        if "images" in row:
            raise ValueError(
                f"row {index} contains source field 'images'; datasets must store exactly "
                "one relative path in 'image' and let EasyR1/EasyQ1 construct images=[image]"
            )

        row_id = str(row["id"]).strip()
        problem_value = row["problem"]
        answer_value = row["answer"]
        problem = problem_value.strip() if isinstance(problem_value, str) else ""
        answer = answer_value.strip() if isinstance(answer_value, str) else ""
        image_value = row["image"]
        if not row_id:
            raise ValueError(f"row {index} has an empty id")
        if row_id in ids:
            raise ValueError(f"row {index} has duplicate id: {row_id}")
        ids.add(row_id)
        if not problem:
            raise ValueError(f"row {index} problem must be a non-empty string")
        if not answer:
            raise ValueError(f"row {index} answer must be a non-empty string")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"row {index} image must be one non-empty relative path string")

        posix_path = PurePosixPath(image_value)
        if posix_path.is_absolute() or ".." in posix_path.parts:
            raise ValueError(f"row {index} has an unsafe image path: {image_value!r}")
        normalized_image = posix_path.as_posix()
        if require_unique_images and normalized_image in images:
            raise ValueError(f"row {index} reuses image path: {normalized_image}")
        images.add(normalized_image)
        image_path = (dataset_dir / Path(*posix_path.parts)).resolve(strict=False)
        try:
            image_path.relative_to(dataset_dir)
        except ValueError as exc:
            raise ValueError(f"row {index} image escapes dataset root: {image_value!r}") from exc
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"row {index} references a missing/empty image: {image_path}")

    return {
        "dataset_dir": str(dataset_dir),
        "rows": len(rows),
        "unique_ids": len(ids),
        "unique_images": len(images),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--allow-shared-images", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_multimodal_dataset(
            args.dataset_dir,
            expected_rows=args.expected_rows,
            require_unique_images=not args.allow_shared_images,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc
    print(
        "Dataset validation OK: "
        f"{result['dataset_dir']} rows={result['rows']} "
        f"unique_ids={result['unique_ids']} unique_images={result['unique_images']} "
        f"sha256={result['data_sha256']}"
    )


if __name__ == "__main__":
    main()
