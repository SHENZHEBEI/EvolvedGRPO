"""Build EasyQ1 prompts for one question-generator training round.

Each row is built from exactly one row of the immediately preceding augmented
dataset.  The output keeps the same cardinality and the same 90% text / 10%
image-edit partition used by ``build_round_k.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from data_augmentation.augment_dataset import (
    MODE_IMAGE_EDIT,
    MODE_TEXT,
    build_agent_prompt,
    load_dataset_rows,
    normalize_relative_path,
    resolve_dataset_image,
    split_rows_for_augmentation,
    transfer_file,
    write_json_atomic,
)


def build_training_rows(
    input_dir: Path,
    output_dir: Path,
    *,
    image_ratio: float,
    seed: int,
    round_name: str,
    transfer: str,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    rows = load_dataset_rows(input_dir)
    if len(rows) < 2:
        raise ValueError("Question-generator training requires at least two rows.")
    text_rows, image_rows = split_rows_for_augmentation(rows, image_ratio, seed)
    mode_by_id = {str(row["id"]): MODE_TEXT for row in text_rows}
    mode_by_id.update({str(row["id"]): MODE_IMAGE_EDIT for row in image_rows})

    output_rows: List[Dict[str, Any]] = []
    transferred = set()
    for source_index, row in enumerate(rows):
        relative_image = normalize_relative_path(row["image"])
        source_image = resolve_dataset_image(input_dir, relative_image)
        destination_image = output_dir / Path(*PurePosixPath(relative_image).parts)
        if relative_image not in transferred:
            transfer_file(source_image, destination_image, transfer, overwrite)
            transferred.add(relative_image)

        mode = mode_by_id[str(row["id"])]
        context = {
            "protocol_version": 1,
            "source_id": str(row["id"]),
            "source_index": source_index,
            "mode": mode,
            "round_name": round_name,
            "previous_problem": str(row["problem"]),
            "previous_answer": str(row["answer"]),
            # The scorer and trainer must see the exact same source image.
            "image_path": str(destination_image.resolve()),
            "image_relative": relative_image,
            "source_dataset": str(input_dir.resolve()),
        }
        output_rows.append(
            {
                "id": f"qgen:{round_name}:{row['id']}",
                "image": relative_image,
                "problem": build_agent_prompt(mode, row),
                # EasyQ1 exposes this field to the custom reward as
                # non_tensor_batch['ground_truth'].
                "answer": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            }
        )

    if len(output_rows) != len(rows):
        raise RuntimeError("Question-generator data changed the per-round row count.")
    write_json_atomic(output_dir / "data.json", output_rows)
    write_json_atomic(
        output_dir / "question_generator_meta.json",
        {
            "protocol_version": 1,
            "round_name": round_name,
            "source_dataset": str(input_dir.resolve()),
            "source_rows": len(rows),
            "output_rows": len(output_rows),
            "text_rows": len(text_rows),
            "image_edit_rows": len(image_rows),
            "image_ratio": image_ratio,
            "seed": seed,
            "transfer": transfer,
            "input_policy": "immediately_previous_round_problem_answer_and_image",
        },
    )
    return output_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-name", required=True)
    parser.add_argument("--image-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    data_path = output_dir / "data.json"
    if data_path.exists() and not args.overwrite:
        raise FileExistsError(f"{data_path} already exists; pass --overwrite to rebuild it.")
    rows = build_training_rows(
        input_dir,
        output_dir,
        image_ratio=args.image_ratio,
        seed=args.seed,
        round_name=args.round_name,
        transfer=args.transfer,
        overwrite=args.overwrite,
    )
    print(f"Question-generator dataset: {output_dir}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

