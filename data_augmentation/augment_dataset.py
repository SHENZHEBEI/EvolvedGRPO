#!/usr/bin/env python3
"""Build a new EasyR1/EasyQ1 training dataset with a Qwen2.5-VL augmentation agent.

The pipeline deliberately separates agent generation, image editing, and dataset
finalization. Each expensive stage writes an append-only manifest, so interrupted
runs can be resumed without modifying the source dataset.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("data_augmentation")

MODE_TEXT = "text"
MODE_IMAGE_EDIT = "image-edit"
SUPPORTED_MODES = (MODE_TEXT, MODE_IMAGE_EDIT)

REQUIRED_DATA_KEYS = ("id", "image", "problem", "answer")
AGENT_MANIFEST_PATTERN = "agent_shard_*_of_*.jsonl"
AGENT_META_PATTERN = "agent_shard_*_of_*.meta.json"
IMAGE_MANIFEST_PATTERN = "image_shard_*_of_*.jsonl"
SOLVER_MANIFEST_PATTERN = "solver_shard_*_of_*.jsonl"

SAFE_EDIT_SUFFIX = (
    " Preserve every original question sentence, numeral, mathematical symbol, formula, table entry, "
    "diagram label, geometric relationship, and answer-relevant detail exactly. Do not cover, erase, "
    "replace, hallucinate, or move any semantic content; only change non-semantic visual appearance."
)

IMAGE_AGENT_PROMPT = """You are an adversarial visual data-augmentation agent for mathematical VQA.
Inspect the image together with the original problem and known answer. Propose one useful image-editing
instruction that makes the visual input harder while leaving the mathematical task and its answer unchanged.

Allowed edits include paper/camera style, illumination, mild perspective, harmless background clutter,
texture, scan artifacts, or annotations only in unused regions.

Forbidden edits: changing, deleting, covering, moving, or inventing any problem text, number, formula,
table value, diagram line, geometric relation, label, option, or answer-relevant object.

Original problem:
{problem}

Known answer (for consistency checking only; do not render it into the image):
{answer}

Return only one JSON object with this exact schema:
{{"edit_instruction": "one precise instruction for an image editing model"}}
"""

TEXT_AGENT_PROMPT = """You are a mathematical problem data-augmentation agent.
Create exactly one concise follow-up mathematical question from the supplied image, previous-round problem,
and previous-round answer. Do not rewrite, summarize, or repeat the previous-round problem: the pipeline will
preserve it verbatim and append your follow-up with a fixed textual bridge.
The known answer below is the answer carried by this sample from the immediately previous training round.
Use it as the construction variable x, then add one concise, solvable follow-up transformation or reasoning
step based on x. For a numeric x, prefer
a simple explicit function such as f(x), a comparison, or a short derived calculation. For a non-numeric or
multiple-choice answer, add a valid conceptual follow-up that can be answered from x and the wording of the
previous problem without inspecting the image again.

Requirements:
1. Keep the previous mathematical facts and the meaning of the image unchanged.
2. Do not introduce any new visual dependency, diagram, label, number, or fact.
3. Refer to the previous answer symbolically as x; never reveal its known value in the follow-up.
4. Output only the follow-up question. Do not include any part of the previous-round problem.
5. Do not output or calculate the new answer. A separate answer model will solve the merged problem with
   multi-round TTRL voting.
6. Make a real augmentation: the follow-up must require a new calculation or reasoning step.

Previous-round problem:
{problem}

Previous-round answer (use to construct the transformation, but never reveal it in the new problem):
{answer}

Return only one JSON object with this exact schema:
{{"follow_up_problem": "one follow-up question expressed in terms of x"}}
"""

TEXT_AGENT_PROMPT_VERSION = "follow_up_then_deterministic_merge_v1"
TEXT_PROBLEM_BRIDGE = (
    "Let x denote the answer to the preceding problem. Using that value, answer the following "
    "connected question:"
)

TEXT_SOLVER_PROMPT = """Solve the new mathematical problem using the previous-round answer as the
authoritative intermediate result x. Do not re-solve the previous visual question and do not require its
image. Apply the transformation or follow-up reasoning requested by the new problem to the supplied answer.
Show concise reasoning, verify the result, and end with exactly one final answer in \\boxed{{}}.
Do not output multiple alternative answers.

Previous-round answer:
{answer}

New augmented problem:
{problem}
"""

TEXT_SOLVER_PROMPT_VERSION = "previous_answer_plus_augmented_problem_v1"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            yield value


def normalize_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid dataset image path: {value!r}")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Dataset image path must stay inside the dataset directory: {value!r}")
    return path.as_posix()


def resolve_dataset_image(dataset_dir: Path, relative_path: Any) -> Path:
    normalized = normalize_relative_path(relative_path)
    candidate = (dataset_dir / Path(*PurePosixPath(normalized).parts)).resolve()
    dataset_root = dataset_dir.resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes dataset directory: {relative_path!r}") from exc
    return candidate


def load_dataset_rows(dataset_dir: Path, require_images: bool = True) -> List[Dict[str, Any]]:
    data_path = dataset_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {data_path}")

    value = read_json(data_path)
    if not isinstance(value, list):
        raise ValueError(f"{data_path} must contain a JSON list.")

    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    missing_images: List[Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset row {index} is not a JSON object.")
        missing_keys = [key for key in REQUIRED_DATA_KEYS if key not in item]
        if missing_keys:
            raise ValueError(f"Dataset row {index} is missing keys: {missing_keys}")

        source_id = str(item["id"])
        if source_id in seen_ids:
            raise ValueError(f"Duplicate dataset id at row {index}: {source_id}")
        seen_ids.add(source_id)

        row = dict(item)
        row["id"] = source_id
        row["image"] = normalize_relative_path(item["image"])
        row["problem"] = str(item["problem"])
        row["answer"] = str(item["answer"])
        rows.append(row)

        if require_images:
            image_path = resolve_dataset_image(dataset_dir, row["image"])
            if not image_path.is_file() and len(missing_images) < 10:
                missing_images.append(image_path)

    if missing_images:
        examples = "\n".join(f"  - {path}" for path in missing_images)
        raise FileNotFoundError(f"Dataset references missing image files. First missing paths:\n{examples}")
    return rows


def resolve_model_path(model: str, config_names: Sequence[str] = ("config.json",)) -> str:
    """Accept a HF model id, a merged HF directory, or an actor directory."""
    candidate = Path(model).expanduser()
    if not candidate.exists():
        return model
    if candidate.is_file():
        raise ValueError(f"Model path must be a directory or Hugging Face model id: {candidate}")
    if any((candidate / config_name).is_file() for config_name in config_names):
        return str(candidate.resolve())
    if (candidate / "huggingface" / "config.json").is_file():
        resolved = (candidate / "huggingface").resolve()
        LOGGER.info("Resolved actor checkpoint to merged Hugging Face directory: %s", resolved)
        return str(resolved)
    raise ValueError(
        f"Local model directory has none of {tuple(config_names)}: {candidate}. "
        "Run scripts/model_merger.py first or pass the actor/huggingface directory."
    )


def shard_paths(work_dir: Path, kind: str, shard_index: int, num_shards: int) -> Tuple[Path, Path]:
    stem = f"{kind}_shard_{shard_index:03d}_of_{num_shards:03d}"
    return work_dir / "manifests" / f"{stem}.jsonl", work_dir / "manifests" / f"{stem}.meta.json"


def validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")


def select_shard(
    rows: Sequence[Dict[str, Any]], max_samples: Optional[int], shard_index: int, num_shards: int
) -> List[Tuple[int, Dict[str, Any]]]:
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        rows = rows[:max_samples]
    return [(index, row) for index, row in enumerate(rows) if index % num_shards == shard_index]


def prepare_stage_files(
    manifest_path: Path,
    meta_path: Path,
    metadata: Dict[str, Any],
    reset_shard: bool,
) -> None:
    if reset_shard:
        for path in (manifest_path, meta_path):
            if path.exists():
                path.unlink()

    if meta_path.exists():
        previous = read_json(meta_path)
        comparison_keys = (
            "stage",
            "mode",
            "input_dir",
            "model",
            "processor_model",
            "max_samples",
            "shard_index",
            "num_shards",
            "settings",
        )
        mismatches = [key for key in comparison_keys if previous.get(key) != metadata.get(key)]
        # Increasing only the solver context is safe for append-only resume:
        # completed consensus answers remain valid and pending long prompts gain
        # more capacity.
        if mismatches == ["settings"] and metadata.get("stage") == "solver":
            previous_settings = dict(previous.get("settings") or {})
            current_settings = dict(metadata.get("settings") or {})
            capacity_keys = {"max_model_len", "max_tokens", "min_pixels", "max_pixels"}
            previous_fixed = {
                key: value
                for key, value in previous_settings.items()
                if key not in capacity_keys
            }
            current_fixed = {
                key: value
                for key, value in current_settings.items()
                if key not in capacity_keys
            }
            previous_context = int(previous_settings.get("max_model_len", 0))
            current_context = int(current_settings.get("max_model_len", 0))
            previous_output = int(previous_settings.get("max_tokens", 0))
            current_output = int(current_settings.get("max_tokens", 0))
            output_change_is_safe = previous_output == 0 or current_output <= previous_output
            if (
                previous_fixed == current_fixed
                and current_context >= previous_context
                and output_change_is_safe
            ):
                LOGGER.warning(
                    "Updating solver capacity (context %d -> %d, output %d -> %d) and "
                    "resuming existing records.",
                    previous_context,
                    current_context,
                    previous_output,
                    current_output,
                )
                updated_metadata = dict(metadata)
                updated_metadata["resumed_from_max_model_len"] = previous_context
                write_json_atomic(meta_path, updated_metadata)
                mismatches = []
        if mismatches:
            raise RuntimeError(
                f"Shard metadata changed for {meta_path}; mismatched keys: {mismatches}. "
                "Use a new work directory or pass --reset-shard."
            )
    else:
        write_json_atomic(meta_path, metadata)


def load_latest_records(paths: Iterable[Path], mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for path in sorted(paths):
        for record in read_jsonl(path):
            if mode is not None and record.get("mode") != mode:
                continue
            source_id = str(record.get("source_id", ""))
            if not source_id:
                raise ValueError(f"Manifest record in {path} has no source_id.")
            latest[source_id] = record
    return latest


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model output does not contain a valid JSON object.")


def compose_text_problem(previous_problem: str, follow_up_problem: str) -> str:
    previous = str(previous_problem).strip()
    follow_up = str(follow_up_problem).strip()
    if not previous or not follow_up:
        raise ValueError("Previous problem and follow-up problem must both be non-empty.")
    return f"{previous}\n\n{TEXT_PROBLEM_BRIDGE}\n{follow_up}"


def agent_input_sha256(mode: str, problem: str, answer: str) -> str:
    prompt_version = TEXT_AGENT_PROMPT_VERSION if mode == MODE_TEXT else "image_edit_v1"
    payload = {
        "mode": mode,
        "prompt_version": prompt_version,
        "problem": str(problem).strip(),
        "answer": str(answer).strip(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_agent_payload(mode: str, raw_output: str, original_problem: str) -> Dict[str, str]:
    try:
        payload = extract_json_object(raw_output)
    except ValueError:
        if mode == MODE_IMAGE_EDIT:
            raise
        # Some trained checkpoints return the requested follow-up directly
        # instead of wrapping it in JSON. Accept the text, but still run every
        # merge/duplication check below.
        plain_text = raw_output.strip()
        plain_text = re.sub(r"^```(?:text|markdown)?\s*", "", plain_text, flags=re.IGNORECASE)
        plain_text = re.sub(r"\s*```$", "", plain_text)
        payload = {"follow_up_problem": plain_text.strip()}
    if mode == MODE_IMAGE_EDIT:
        instruction = payload.get("edit_instruction", payload.get("instruction", payload.get("prompt")))
        if not isinstance(instruction, str) or len(instruction.strip()) < 10:
            raise ValueError("Image agent output has no usable edit_instruction.")
        return {"edit_instruction": instruction.strip()}

    follow_up = payload.get(
        "follow_up_problem",
        payload.get("follow_up_question", payload.get("problem")),
    )
    if not isinstance(follow_up, str) or not follow_up.strip():
        raise ValueError("Text agent output has no usable follow_up_problem.")
    follow_up = follow_up.strip()
    previous = original_problem.strip()
    if previous in follow_up:
        # Salvage checkpoints that return a complete combined problem: discard
        # their copy of the previous problem, keep only the appended part, and
        # rebuild the final problem using our fixed bridge.
        follow_up = follow_up.split(previous, 1)[1].strip()
    if TEXT_PROBLEM_BRIDGE in follow_up:
        follow_up = follow_up.split(TEXT_PROBLEM_BRIDGE, 1)[1].strip()
    follow_up = re.sub(
        r"^(?:follow[- ]?up (?:problem|question)|new (?:problem|question))\s*:\s*",
        "",
        follow_up,
        flags=re.IGNORECASE,
    ).strip()
    follow_up = follow_up.lstrip(":- \t\r\n")
    if not follow_up or follow_up == previous:
        raise ValueError("Text agent did not produce a distinct follow-up problem.")
    result = {
        "follow_up_problem": follow_up,
        "problem": compose_text_problem(original_problem, follow_up),
    }
    proposed_answer = payload.get("answer", payload.get("augmented_answer"))
    if isinstance(proposed_answer, (str, int, float)) and str(proposed_answer).strip():
        result["agent_proposed_answer"] = str(proposed_answer).strip()
    return result


def problem_sha256(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()


def solver_input_sha256(problem: str, previous_answer: str) -> str:
    payload = {
        "prompt_version": TEXT_SOLVER_PROMPT_VERSION,
        "problem": str(problem).strip(),
        "previous_answer": str(previous_answer).strip(),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_boxed_answer(text: str, mathruler_extractor: Optional[Any] = None) -> Optional[str]:
    if mathruler_extractor is not None:
        try:
            answer = mathruler_extractor(text)
            if answer is not None and str(answer).strip().lower() not in ("", "none"):
                return str(answer).strip()
        except Exception:
            pass

    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = text[content_start:index].strip()
                return answer or None
    return None


def canonicalize_answer(answer: str) -> str:
    value = str(answer).strip()
    while len(value) >= 2 and value[0] == "$" and value[-1] == "$":
        value = value[1:-1].strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "")
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def grade_result_is_true(result: Any) -> bool:
    if isinstance(result, tuple):
        result = result[0] if result else False
    if isinstance(result, dict):
        for key in ("correct", "is_correct", "result"):
            if key in result:
                return bool(result[key])
    return bool(result)


def answers_equivalent(left: str, right: str, grade_answer_fn: Optional[Any] = None) -> bool:
    if canonicalize_answer(left) == canonicalize_answer(right):
        return True
    if grade_answer_fn is None:
        return False
    for candidate, reference in ((left, right), (right, left)):
        try:
            if grade_result_is_true(grade_answer_fn(candidate, reference)):
                return True
        except Exception:
            continue
    return False


def summarize_votes(
    answers: Sequence[str],
    grade_answer_fn: Optional[Any] = None,
    total_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    for answer in answers:
        for cluster in clusters:
            if answers_equivalent(answer, cluster["answer"], grade_answer_fn):
                cluster["members"].append(answer)
                break
        else:
            clusters.append({"answer": answer, "members": [answer]})

    clusters.sort(
        key=lambda cluster: (-len(cluster["members"]), canonicalize_answer(cluster["answer"]))
    )
    valid_votes = len(answers)
    sampled_candidates = total_candidates if total_candidates is not None else valid_votes
    if sampled_candidates < valid_votes:
        raise ValueError("total_candidates cannot be smaller than the number of valid answers.")
    if not clusters:
        return {
            "answer": None,
            "winning_votes": 0,
            "valid_votes": 0,
            "confidence": 0.0,
            "candidate_share": 0.0,
            "sampled_candidates": sampled_candidates,
            "vote_counts": [],
        }
    winner = clusters[0]
    winning_votes = len(winner["members"])
    return {
        "answer": winner["answer"],
        "winning_votes": winning_votes,
        "valid_votes": valid_votes,
        "confidence": winning_votes / valid_votes,
        "candidate_share": winning_votes / sampled_candidates if sampled_candidates else 0.0,
        "sampled_candidates": sampled_candidates,
        "vote_counts": [
            {"answer": cluster["answer"], "count": len(cluster["members"])}
            for cluster in clusters
        ],
    }


def build_agent_prompt(mode: str, row: Dict[str, Any]) -> str:
    template = IMAGE_AGENT_PROMPT if mode == MODE_IMAGE_EDIT else TEXT_AGENT_PROMPT
    return template.format(problem=row["problem"], answer=row["answer"])


def build_vllm_input(
    processor: Any,
    process_vision_info: Any,
    image_path: Path,
    prompt_text: str,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    image_content: Dict[str, Any] = {"type": "image", "image": str(image_path)}
    if min_pixels is not None:
        image_content["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_content["max_pixels"] = max_pixels
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_data, video_data = process_vision_info(messages)
    multi_modal_data: Dict[str, Any] = {}
    if image_data is not None:
        multi_modal_data["image"] = image_data
    if video_data is not None:
        multi_modal_data["video"] = video_data
    return {"prompt": prompt, "multi_modal_data": multi_modal_data}


def build_text_vllm_input(processor: Any, prompt_text: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt}


def generate_agent_outputs(args: argparse.Namespace) -> None:
    validate_shard_args(args.shard_index, args.num_shards)
    if args.max_model_len <= 0 or args.max_tokens <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError(
            "--max-model-len, --max-tokens and --max-num-batched-tokens must be positive."
        )
    if args.max_generation_attempts <= 0:
        raise ValueError("--max-generation-attempts must be positive.")
    if args.min_pixels <= 0 or args.max_pixels <= 0 or args.min_pixels > args.max_pixels:
        raise ValueError("--min-pixels and --max-pixels must satisfy 0 < min <= max.")
    input_dir = Path(args.input_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    rows = load_dataset_rows(input_dir)
    selected = select_shard(rows, args.max_samples, args.shard_index, args.num_shards)

    agent_model = resolve_model_path(args.agent_model)
    processor_model = resolve_model_path(args.processor_model or args.agent_model)
    manifest_path, meta_path = shard_paths(work_dir, "agent", args.shard_index, args.num_shards)
    metadata = {
        "stage": "agent",
        "mode": args.mode,
        "input_dir": str(input_dir),
        "model": agent_model,
        "processor_model": processor_model,
        "max_samples": args.max_samples,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_count": len(selected),
        "dataset_count": len(rows),
        "seed": args.seed,
        "prompt_version": (
            TEXT_AGENT_PROMPT_VERSION if args.mode == MODE_TEXT else "image_edit_v1"
        ),
        "input_capacity": {
            "max_model_len": args.max_model_len,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "max_output_tokens": args.max_tokens,
            "max_generation_attempts": args.max_generation_attempts,
        },
    }
    prepare_stage_files(manifest_path, meta_path, metadata, args.reset_shard)

    existing = load_latest_records([manifest_path], mode=args.mode) if manifest_path.exists() else {}
    recovered_records: List[Dict[str, Any]] = []
    if args.mode == MODE_TEXT:
        for source_index, row in selected:
            previous = existing.get(row["id"], {})
            current_input_hash = agent_input_sha256(
                args.mode, str(row["problem"]), str(row["answer"])
            )
            if (
                previous.get("status") != "success"
                and previous.get("agent_input_sha256") == current_input_hash
                and isinstance(previous.get("raw_output"), str)
            ):
                try:
                    normalized = normalize_agent_payload(
                        args.mode, str(previous["raw_output"]), str(row["problem"])
                    )
                except Exception:
                    continue
                recovered = dict(previous)
                recovered.update(normalized)
                recovered["status"] = "success"
                recovered["source_index"] = source_index
                recovered["recovered_without_regeneration"] = True
                recovered.pop("error", None)
                recovered_records.append(recovered)
                existing[row["id"]] = recovered
        if recovered_records:
            append_jsonl(manifest_path, recovered_records)
            LOGGER.info(
                "Recovered %d text Agent records from existing raw outputs without inference.",
                len(recovered_records),
            )

    pending = []
    reusable_count = 0
    terminal_fallback_count = 0
    for source_index, row in selected:
        previous = existing.get(row["id"], {})
        current_input_hash = agent_input_sha256(
            args.mode, str(row["problem"]), str(row["answer"])
        )
        reusable = previous.get("status") == "success"
        if args.mode == MODE_TEXT:
            reusable = reusable and previous.get("agent_input_sha256") == current_input_hash
        if reusable:
            reusable_count += 1
            continue
        if (
            previous.get("status") == "error"
            and previous.get("agent_input_sha256") == current_input_hash
        ):
            terminal_fallback_count += 1
            continue
        pending.append((source_index, row))
    LOGGER.info(
        "Agent shard %d/%d: selected=%d, reusable=%d, unchanged fallback=%d, pending=%d",
        args.shard_index,
        args.num_shards,
        len(selected),
        reusable_count,
        terminal_fallback_count,
        len(pending),
    )
    if not pending:
        return

    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "Agent generation requires vllm, transformers, and qwen-vl-utils. "
            f"Import failed with {exc!r} under {sys.executable}. Install and validate the "
            f"pinned CUDA runtime with: source {Path(__file__).resolve().parent.parent / 'augmentation_env.sh'}"
        ) from exc

    processor = AutoProcessor.from_pretrained(processor_model, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=agent_model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    failed_after_retries = 0
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        model_inputs: List[Dict[str, Any]] = []
        prepared: List[Tuple[int, Dict[str, Any]]] = []
        immediate_records: List[Dict[str, Any]] = []

        for source_index, row in batch:
            try:
                image_path = resolve_dataset_image(input_dir, row["image"])
                model_input = build_vllm_input(
                    processor,
                    process_vision_info,
                    image_path,
                    build_agent_prompt(args.mode, row),
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError("Agent processor has no tokenizer for prompt length checking.")
                text_prompt_tokens = len(
                    tokenizer.encode(model_input["prompt"], add_special_tokens=False)
                )
                maximum_visual_tokens = (args.max_pixels + 28 * 28 - 1) // (28 * 28)
                estimated_total_tokens = (
                    text_prompt_tokens + maximum_visual_tokens + args.max_tokens
                )
                if estimated_total_tokens > args.max_model_len:
                    raise ValueError(
                        "bounded Agent prompt plus output can require up to "
                        f"{estimated_total_tokens} tokens, exceeding "
                        f"max_model_len={args.max_model_len}"
                    )
                model_inputs.append(model_input)
                prepared.append((source_index, row))
            except Exception as exc:
                immediate_records.append(
                    {
                        "source_id": row["id"],
                        "source_index": source_index,
                        "mode": args.mode,
                        "status": "error",
                        "agent_input_sha256": agent_input_sha256(
                            args.mode, str(row["problem"]), str(row["answer"])
                        ),
                        "error": f"input preparation failed: {exc}",
                    }
                )

        if immediate_records:
            append_jsonl(manifest_path, immediate_records)
            failed_after_retries += len(immediate_records)
        if not model_inputs:
            continue

        records: List[Dict[str, Any]] = []
        retry_offset = max(
            (
                int(existing.get(row["id"], {}).get("generation_attempts", 0) or 0)
                for _, row in prepared
            ),
            default=0,
        )
        raw_outputs_by_index: Dict[int, List[str]] = {
            index: [] for index in range(len(prepared))
        }
        validation_errors: Dict[int, List[str]] = {
            index: [] for index in range(len(prepared))
        }
        active = list(range(len(prepared)))
        for attempt in range(1, args.max_generation_attempts + 1):
            if not active:
                break
            sampling_params = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + args.shard_index * 10000 + retry_offset + attempt,
            )
            try:
                outputs = llm.generate(
                    [model_inputs[index] for index in active],
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )
            except Exception as exc:
                append_jsonl(
                    manifest_path,
                    [
                        {
                            "source_id": prepared[index][1]["id"],
                            "source_index": prepared[index][0],
                            "mode": args.mode,
                            "status": "error",
                            "error": f"vLLM generation failed on attempt {attempt}: {exc}",
                        }
                        for index in active
                    ],
                )
                raise

            next_active: List[int] = []
            for prepared_index, output in zip(active, outputs):
                source_index, row = prepared[prepared_index]
                raw_output = output.outputs[0].text if output.outputs else ""
                raw_outputs_by_index[prepared_index].append(raw_output)
                record: Dict[str, Any] = {
                    "source_id": row["id"],
                    "source_index": source_index,
                    "mode": args.mode,
                    "status": "success",
                    "agent_input_sha256": agent_input_sha256(
                        args.mode, str(row["problem"]), str(row["answer"])
                    ),
                    # Record the exact previous-round inputs used by the Agent.
                    "previous_problem_sha256": problem_sha256(row["problem"]),
                    "previous_answer": row["answer"],
                    "raw_output": raw_output,
                    "raw_outputs": list(raw_outputs_by_index[prepared_index]),
                    "generation_attempts": retry_offset + attempt,
                    "attempts_this_run": attempt,
                    "agent_model": agent_model,
                }
                try:
                    record.update(
                        normalize_agent_payload(args.mode, raw_output, row["problem"])
                    )
                except Exception as exc:
                    validation_errors[prepared_index].append(str(exc))
                    if attempt < args.max_generation_attempts:
                        next_active.append(prepared_index)
                        continue
                    record["status"] = "error"
                    record["error"] = (
                        "output validation failed after "
                        f"{attempt} attempts: {validation_errors[prepared_index]}"
                    )
                    failed_after_retries += 1
                records.append(record)
            active = next_active

        append_jsonl(manifest_path, records)
        LOGGER.info("Agent shard %d wrote %d/%d pending records.", args.shard_index, min(batch_start + len(batch), len(pending)), len(pending))

    if failed_after_retries:
        message = (
            f"Agent shard {args.shard_index} has {failed_after_retries} failed records "
            f"after {args.max_generation_attempts} attempt(s)."
        )
        LOGGER.warning("%s They will be preserved unchanged in the new round.", message)


def solve_text_answers(args: argparse.Namespace) -> None:
    validate_shard_args(args.shard_index, args.num_shards)
    if args.votes_per_round <= 0 or args.max_vote_rounds <= 0:
        raise ValueError("--votes-per-round and --max-vote-rounds must be positive.")
    if args.min_vote_rounds <= 0 or args.min_vote_rounds > args.max_vote_rounds:
        raise ValueError("--min-vote-rounds must satisfy 1 <= min <= max vote rounds.")
    if args.min_valid_votes <= 0 or args.min_agree_votes <= 0:
        raise ValueError("--min-valid-votes and --min-agree-votes must be positive.")
    maximum_votes = args.votes_per_round * args.max_vote_rounds
    if args.min_valid_votes > maximum_votes or args.min_agree_votes > maximum_votes:
        raise ValueError(
            "--min-valid-votes and --min-agree-votes cannot exceed the maximum sampled votes."
        )
    if not 0.0 < args.consensus_threshold <= 1.0:
        raise ValueError("--consensus-threshold must be in (0, 1].")
    if args.batch_size <= 0 or args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError(
            "--batch-size, --max-num-seqs and --max-num-batched-tokens must be positive."
        )
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive.")
    input_dir = Path(args.input_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    rows = load_dataset_rows(input_dir)
    rows_by_id = {row["id"]: row for row in rows}

    agent_records = load_latest_records(
        (work_dir / "manifests").glob(AGENT_MANIFEST_PATTERN), MODE_TEXT
    )
    selected = [
        record
        for record in agent_records.values()
        if record.get("status") == "success"
        and int(record["source_index"]) % args.num_shards == args.shard_index
    ]
    selected.sort(key=lambda record: int(record["source_index"]))

    solver_model = resolve_model_path(args.solver_model)
    processor_model = resolve_model_path(args.processor_model or args.solver_model)
    manifest_path, meta_path = shard_paths(work_dir, "solver", args.shard_index, args.num_shards)
    settings = {
        "votes_per_round": args.votes_per_round,
        "min_vote_rounds": args.min_vote_rounds,
        "max_vote_rounds": args.max_vote_rounds,
        "min_valid_votes": args.min_valid_votes,
        "min_agree_votes": args.min_agree_votes,
        "consensus_threshold": args.consensus_threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
    }
    metadata = {
        "stage": "solver",
        "mode": MODE_TEXT,
        "input_dir": str(input_dir),
        "model": solver_model,
        "processor_model": processor_model,
        "max_samples": None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_count": len(selected),
        "dataset_count": len(rows),
        "seed": args.seed,
        "prompt_version": TEXT_SOLVER_PROMPT_VERSION,
        "settings": settings,
    }
    prepare_stage_files(manifest_path, meta_path, metadata, args.reset_shard)

    existing = load_latest_records([manifest_path], MODE_TEXT) if manifest_path.exists() else {}
    pending = []
    for record in selected:
        source_row = rows_by_id.get(str(record["source_id"]))
        if source_row is None:
            pending.append(record)
            continue
        current_hash = solver_input_sha256(
            str(record["problem"]), str(source_row["answer"])
        )
        previous = existing.get(record["source_id"])
        if (
            previous
            and previous.get("status") == "success"
            and previous.get("solver_input_sha256") == current_hash
        ):
            continue
        pending.append(record)

    LOGGER.info(
        "TTRL solver shard %d/%d: selected=%d, already successful=%d, pending=%d",
        args.shard_index,
        args.num_shards,
        len(selected),
        len(selected) - len(pending),
        len(pending),
    )
    if not pending:
        return

    try:
        from mathruler.grader import extract_boxed_content, grade_answer
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "TTRL answer voting requires mathruler, vllm, and transformers. "
            f"Import failed with {exc!r} under {sys.executable}. Install and validate the "
            f"pinned CUDA runtime with: source {Path(__file__).resolve().parent.parent / 'augmentation_env.sh'}"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        processor_model, trust_remote_code=args.trust_remote_code
    )
    llm = LLM(
        model=solver_model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        prepared: List[Dict[str, Any]] = []
        immediate_records: List[Dict[str, Any]] = []
        for agent_record in batch:
            source_id = str(agent_record["source_id"])
            source_row = rows_by_id.get(source_id)
            if source_row is None:
                immediate_records.append(
                    {
                        "source_id": source_id,
                        "source_index": int(agent_record["source_index"]),
                        "mode": MODE_TEXT,
                        "status": "error",
                        "error": "source id no longer exists in input dataset",
                    }
                )
                continue
            problem = str(agent_record["problem"])
            current_solver_input_hash = solver_input_sha256(
                problem, str(source_row["answer"])
            )
            try:
                model_input = build_text_vllm_input(
                    processor,
                    TEXT_SOLVER_PROMPT.format(
                        problem=problem,
                        answer=source_row["answer"],
                    ),
                )
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise RuntimeError("Solver processor has no tokenizer for prompt length checking.")
                prompt_tokens = len(
                    tokenizer.encode(model_input["prompt"], add_special_tokens=False)
                )
                required_tokens = prompt_tokens + args.max_tokens
                if required_tokens > args.max_model_len:
                    raise ValueError(
                        f"text-only TTRL requires {prompt_tokens} prompt + {args.max_tokens} "
                        f"output tokens, exceeding max_model_len={args.max_model_len}"
                    )
            except Exception as exc:
                immediate_records.append(
                    {
                        "source_id": source_id,
                        "source_index": int(agent_record["source_index"]),
                        "mode": MODE_TEXT,
                        "status": "error",
                        "problem_sha256": problem_sha256(problem),
                        "solver_input_sha256": current_solver_input_hash,
                        "error": f"solver input preparation failed: {exc}",
                    }
                )
                continue
            prepared.append(
                {
                    "source_id": source_id,
                    "source_index": int(agent_record["source_index"]),
                    "problem": problem,
                    "problem_sha256": problem_sha256(problem),
                    "solver_input_sha256": current_solver_input_hash,
                    "prompt_tokens": prompt_tokens,
                    "model_input": model_input,
                    "answers": [],
                    "raw_outputs": [],
                    "expected_candidates": 0,
                    "returned_candidates": 0,
                    "rounds_completed": 0,
                    "summary": summarize_votes([], total_candidates=0),
                    "resolved": False,
                }
            )

        if immediate_records:
            append_jsonl(manifest_path, immediate_records)
        if not prepared:
            continue

        active = list(range(len(prepared)))
        for vote_round in range(1, args.max_vote_rounds + 1):
            if not active:
                break
            sampling_params = SamplingParams(
                n=args.votes_per_round,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + args.shard_index * 10000 + vote_round,
            )
            round_inputs = [prepared[index]["model_input"] for index in active]
            try:
                outputs = llm.generate(
                    round_inputs,
                    sampling_params=sampling_params,
                    use_tqdm=True,
                )
            except Exception as exc:
                append_jsonl(
                    manifest_path,
                    [
                        {
                            "source_id": prepared[index]["source_id"],
                            "source_index": prepared[index]["source_index"],
                            "mode": MODE_TEXT,
                            "status": "error",
                            "problem_sha256": prepared[index]["problem_sha256"],
                            "solver_input_sha256": prepared[index]["solver_input_sha256"],
                            "error": f"TTRL vLLM generation failed: {exc}",
                        }
                        for index in active
                    ],
                )
                raise

            next_active = []
            for prepared_index, output in zip(active, outputs):
                state = prepared[prepared_index]
                round_raw_outputs = [candidate.text for candidate in output.outputs]
                round_answers = [
                    answer
                    for answer in (
                        extract_boxed_answer(raw_output, extract_boxed_content)
                        for raw_output in round_raw_outputs
                    )
                    if answer is not None
                ]
                state["answers"].extend(round_answers)
                state["expected_candidates"] += args.votes_per_round
                state["returned_candidates"] += len(round_raw_outputs)
                if args.store_raw_outputs:
                    state["raw_outputs"].extend(round_raw_outputs)
                state["rounds_completed"] = vote_round
                state["summary"] = summarize_votes(
                    state["answers"],
                    grade_answer,
                    total_candidates=state["expected_candidates"],
                )

                summary = state["summary"]
                state["resolved"] = (
                    vote_round >= args.min_vote_rounds
                    and summary["valid_votes"] >= args.min_valid_votes
                    and summary["winning_votes"] >= args.min_agree_votes
                    and summary["candidate_share"] >= args.consensus_threshold
                )
                if not state["resolved"]:
                    next_active.append(prepared_index)
            active = next_active

        solver_records: List[Dict[str, Any]] = []
        for state in prepared:
            summary = state["summary"]
            fallback_to_source = not state["resolved"]
            source_row = rows_by_id[state["source_id"]]
            record = {
                "source_id": state["source_id"],
                "source_index": state["source_index"],
                "mode": MODE_TEXT,
                "status": "success",
                "problem_sha256": state["problem_sha256"],
                "solver_input_sha256": state["solver_input_sha256"],
                "solver_model": solver_model,
                "prompt_tokens": state["prompt_tokens"],
                # The previous answer was used by the question Agent to build
                # the transformed problem. It is not the new training label.
                "previous_answer": source_row["answer"],
                "answer": source_row["answer"] if fallback_to_source else summary["answer"],
                "fallback_to_source": fallback_to_source,
                "candidate_answers": state["answers"],
                "vote_counts": summary["vote_counts"],
                "valid_votes": summary["valid_votes"],
                "winning_votes": summary["winning_votes"],
                "confidence": summary["confidence"],
                "candidate_share": summary["candidate_share"],
                "expected_candidates": state["expected_candidates"],
                "returned_candidates": state["returned_candidates"],
                "rounds_completed": state["rounds_completed"],
            }
            if args.store_raw_outputs:
                record["raw_outputs"] = state["raw_outputs"]
            if fallback_to_source:
                record["fallback_reason"] = (
                    "No answer cluster reached the configured share; preserve the source row."
                )
            solver_records.append(record)
        append_jsonl(manifest_path, solver_records)
        LOGGER.info(
            "TTRL solver shard %d wrote %d/%d pending records.",
            args.shard_index,
            min(batch_start + len(batch), len(pending)),
            len(pending),
        )


def stable_token(source_id: str, mode: str, round_name: str) -> str:
    value = f"easy-augmentation:{round_name}:{mode}:{source_id}"
    return uuid.uuid5(uuid.NAMESPACE_URL, value).hex


def save_image_atomic(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_qwen_image_edit_pipeline_compat(
    pipeline_class: Any,
    image_model: str,
    torch_dtype: Any,
    generation_config_class: Optional[Any] = None,
) -> Any:
    """Load Qwen-Image-Edit-2511 without upgrading the vLLM Transformers pin.

    Qwen-Image-Edit-2511 currently ships a Qwen2.5-VL text-encoder config
    serialized by Transformers 4.57.1. Transformers 4.51.3, which is retained
    for the vLLM 0.8.3 runtime, leaves its nested ``text_config`` as a plain
    dict and then incorrectly calls ``to_dict()`` on it while constructing a
    GenerationConfig. All generation fields required here are also present on
    the outer config, so temporarily hiding only that incompatible dict during
    model construction is equivalent to loading the older flat Qwen2.5-VL
    config. The original class method and model config are restored immediately.
    """
    if generation_config_class is None:
        from transformers.generation.configuration_utils import GenerationConfig

        generation_config_class = GenerationConfig

    original_descriptor = generation_config_class.__dict__.get("from_model_config")
    if not isinstance(original_descriptor, classmethod):
        return pipeline_class.from_pretrained(image_model, torch_dtype=torch_dtype)
    original_function = original_descriptor.__func__

    @classmethod
    def compatible_from_model_config(cls: Any, model_config: Any) -> Any:
        nested_text_config = getattr(model_config, "text_config", None)
        if not isinstance(nested_text_config, dict):
            return original_function(cls, model_config)
        LOGGER.warning(
            "Applying the Transformers 4.51 compatibility path for the "
            "Qwen-Image-Edit-2511 nested text_config."
        )
        delattr(model_config, "text_config")
        try:
            return original_function(cls, model_config)
        finally:
            setattr(model_config, "text_config", nested_text_config)

    generation_config_class.from_model_config = compatible_from_model_config
    try:
        return pipeline_class.from_pretrained(image_model, torch_dtype=torch_dtype)
    finally:
        generation_config_class.from_model_config = original_descriptor


def preserve_pending_images_after_stage_failure(
    manifest_path: Path,
    pending: Sequence[Dict[str, Any]],
    error: str,
) -> None:
    append_jsonl(
        manifest_path,
        [
            {
                "source_id": record["source_id"],
                "source_index": int(record["source_index"]),
                "mode": MODE_IMAGE_EDIT,
                "status": "error",
                "error": error,
                "fallback_to_source": True,
            }
            for record in pending
        ],
    )


def edit_images(args: argparse.Namespace) -> None:
    validate_shard_args(args.shard_index, args.num_shards)
    input_dir = Path(args.input_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    rows = load_dataset_rows(input_dir)
    rows_by_id = {row["id"]: row for row in rows}

    agent_records = load_latest_records((work_dir / "manifests").glob(AGENT_MANIFEST_PATTERN), MODE_IMAGE_EDIT)
    successful_agent_records = {
        source_id: record for source_id, record in agent_records.items() if record.get("status") == "success"
    }
    selected = [
        record
        for record in successful_agent_records.values()
        if int(record["source_index"]) % args.num_shards == args.shard_index
    ]
    selected.sort(key=lambda record: int(record["source_index"]))

    image_model = resolve_model_path(args.image_model, ("model_index.json", "config.json"))
    manifest_path, meta_path = shard_paths(work_dir, "image", args.shard_index, args.num_shards)
    metadata = {
        "stage": "image",
        "mode": MODE_IMAGE_EDIT,
        "input_dir": str(input_dir),
        "model": image_model,
        "processor_model": None,
        "max_samples": None,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_count": len(selected),
        "seed": args.seed,
    }
    prepare_stage_files(manifest_path, meta_path, metadata, args.reset_shard)
    existing = load_latest_records([manifest_path], MODE_IMAGE_EDIT) if manifest_path.exists() else {}

    pending: List[Dict[str, Any]] = []
    for record in selected:
        current = existing.get(record["source_id"])
        if current and current.get("status") == "success":
            edited_file = work_dir / str(current.get("edited_file", ""))
            if edited_file.is_file():
                continue
        pending.append(record)

    LOGGER.info(
        "Image shard %d/%d: selected=%d, already successful=%d, pending=%d",
        args.shard_index,
        args.num_shards,
        len(selected),
        len(selected) - len(pending),
        len(pending),
    )
    LOGGER.info(
        "Image inference settings: steps=%d, true_cfg_scale=%s, guidance_scale=%s, "
        "internal resolution=approximately 1MP, preserve source size=%s",
        args.num_inference_steps,
        args.true_cfg_scale,
        args.guidance_scale,
        args.preserve_size,
    )
    if not pending:
        return

    try:
        import torch
        from PIL import Image
        from diffusers import QwenImageEditPlusPipeline
    except ImportError as exc:
        message = (
            "Qwen-Image-Edit-2511 requires a recent diffusers build exposing "
            f"QwenImageEditPlusPipeline. Import failed with {exc!r} under {sys.executable}. "
            f"Install and validate the pinned CUDA runtime with: source "
            f"{Path(__file__).resolve().parent.parent / 'augmentation_env.sh'}"
        )
        LOGGER.exception("%s Preserving all %d pending source images.", message, len(pending))
        preserve_pending_images_after_stage_failure(manifest_path, pending, message)
        return

    try:
        pipeline = load_qwen_image_edit_pipeline_compat(
            QwenImageEditPlusPipeline,
            image_model,
            torch.bfloat16,
        )
        if args.cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(args.device)
        pipeline.set_progress_bar_config(disable=False)
    except Exception as exc:
        message = f"Qwen image pipeline initialization failed: {type(exc).__name__}: {exc}"
        LOGGER.exception("%s Preserving all %d pending source images.", message, len(pending))
        preserve_pending_images_after_stage_failure(manifest_path, pending, message)
        return

    for offset, agent_record in enumerate(pending, start=1):
        source_id = agent_record["source_id"]
        source_index = int(agent_record["source_index"])
        row = rows_by_id.get(source_id)
        if row is None:
            append_jsonl(
                manifest_path,
                [
                    {
                        "source_id": source_id,
                        "source_index": source_index,
                        "mode": MODE_IMAGE_EDIT,
                        "status": "error",
                        "error": "source id no longer exists in input dataset",
                    }
                ],
            )
            continue

        token = stable_token(source_id, MODE_IMAGE_EDIT, args.round_name)
        edited_relative = PurePosixPath("edited_images", f"{token}.png").as_posix()
        edited_path = work_dir / "edited_images" / f"{token}.png"
        instruction = str(agent_record["edit_instruction"]).strip() + SAFE_EDIT_SUFFIX
        try:
            input_path = resolve_dataset_image(input_dir, row["image"])
            with Image.open(input_path) as opened:
                input_image = opened.convert("RGB")
                original_size = input_image.size

            generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(args.seed + source_index)
            with torch.inference_mode():
                result = pipeline(
                    image=[input_image],
                    prompt=instruction,
                    generator=generator,
                    true_cfg_scale=args.true_cfg_scale,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    num_images_per_prompt=1,
                )
            output_image = result.images[0].convert("RGB")
            if args.preserve_size and output_image.size != original_size:
                output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)
            save_image_atomic(output_image, edited_path)
            record = {
                "source_id": source_id,
                "source_index": source_index,
                "mode": MODE_IMAGE_EDIT,
                "status": "success",
                "edited_file": edited_relative,
                "edit_instruction": instruction,
                "image_model": image_model,
                "num_inference_steps": args.num_inference_steps,
                "true_cfg_scale": args.true_cfg_scale,
                "guidance_scale": args.guidance_scale,
            }
        except Exception as exc:
            record = {
                "source_id": source_id,
                "source_index": source_index,
                "mode": MODE_IMAGE_EDIT,
                "status": "error",
                "error": str(exc),
            }
            LOGGER.exception("Image editing failed for source id %s", source_id)
        append_jsonl(manifest_path, [record])
        LOGGER.info("Image shard %d completed %d/%d pending images.", args.shard_index, offset, len(pending))


def validate_agent_metadata(work_dir: Path, mode: str, allow_partial: bool) -> int:
    meta_paths = sorted((work_dir / "manifests").glob(AGENT_META_PATTERN))
    if not meta_paths:
        raise FileNotFoundError(f"No agent shard metadata found under {work_dir / 'manifests'}")

    metadata = []
    for path in meta_paths:
        item = read_json(path)
        if item.get("mode") == mode:
            metadata.append(item)
    if not metadata:
        raise FileNotFoundError(f"No agent shard metadata found for mode {mode!r}.")

    num_shards_values = {int(item["num_shards"]) for item in metadata}
    if len(num_shards_values) != 1:
        raise ValueError(f"Inconsistent agent num_shards values: {num_shards_values}")
    num_shards = num_shards_values.pop()
    shard_indices = {int(item["shard_index"]) for item in metadata}
    missing_shards = sorted(set(range(num_shards)) - shard_indices)
    if missing_shards and not allow_partial:
        raise RuntimeError(f"Agent stage is missing shard metadata for shards: {missing_shards}")
    return sum(int(item["selected_count"]) for item in metadata)


def transfer_file(source: Path, destination: Path, method: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and files_identical(source, destination):
            return
        if not overwrite:
            raise FileExistsError(
                f"Destination exists with different content: {destination}. Pass --overwrite to replace it."
            )
        destination.unlink()

    if method == "copy":
        shutil.copy2(source, destination)
        return
    if method == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            LOGGER.warning("Hardlink failed for %s; falling back to a file copy.", source)
            shutil.copy2(source, destination)
        return
    if method == "symlink":
        destination.symlink_to(source.resolve())
        return
    raise ValueError(f"Unknown transfer method: {method}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def files_identical(left: Path, right: Path) -> bool:
    try:
        if left.samefile(right):
            return True
    except (FileNotFoundError, OSError):
        pass
    if not left.is_file() or not right.is_file():
        return False
    if left.stat().st_size != right.stat().st_size:
        return False
    return file_sha256(left) == file_sha256(right)


def build_augmented_id(source_id: str, mode: str, round_name: str) -> str:
    suffix = stable_token(source_id, mode, round_name)[:12]
    return f"{source_id}__{mode.replace('-', '_')}__{round_name}__{suffix}"


def finalize_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == input_dir:
        raise ValueError("Refusing to overwrite the source dataset. Choose a different --output-dir.")

    output_data_path = output_dir / "data.json"
    if output_data_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_data_path} already exists. Pass --overwrite to rebuild it.")

    rows = load_dataset_rows(input_dir)
    rows_by_id = {row["id"]: row for row in rows}
    expected = validate_agent_metadata(work_dir, args.mode, args.allow_partial)
    agent_records = load_latest_records((work_dir / "manifests").glob(AGENT_MANIFEST_PATTERN), args.mode)
    agent_success = {
        source_id: record for source_id, record in agent_records.items() if record.get("status") == "success"
    }
    agent_failed = {
        source_id: record for source_id, record in agent_records.items() if record.get("status") != "success"
    }
    if len(agent_records) < expected and not args.allow_partial:
        raise RuntimeError(f"Agent stage has {len(agent_records)} records but metadata expects {expected}.")
    solver_success: Dict[str, Dict[str, Any]] = {}
    if args.mode == MODE_TEXT:
        solver_records = load_latest_records(
            (work_dir / "manifests").glob(SOLVER_MANIFEST_PATTERN), MODE_TEXT
        )
        for source_id, solver_record in solver_records.items():
            agent_record = agent_success.get(source_id)
            if agent_record is None or solver_record.get("status") != "success":
                continue
            source_row = rows_by_id.get(source_id)
            if source_row is None:
                continue
            expected_solver_hash = solver_input_sha256(
                str(agent_record["problem"]), str(source_row["answer"])
            )
            if solver_record.get("solver_input_sha256") != expected_solver_hash:
                continue
            solver_success[source_id] = solver_record
        missing_solver_results = sorted(set(agent_success) - set(solver_success))
        if missing_solver_results and not args.allow_partial:
            raise RuntimeError(
                f"TTRL solving is missing {len(missing_solver_results)} consensus answers. "
                f"First missing source id: {missing_solver_results[0]}"
            )

    image_success: Dict[str, Dict[str, Any]] = {}
    if args.mode == MODE_IMAGE_EDIT:
        image_records = load_latest_records((work_dir / "manifests").glob(IMAGE_MANIFEST_PATTERN), MODE_IMAGE_EDIT)
        image_success = {
            source_id: record for source_id, record in image_records.items() if record.get("status") == "success"
        }

    output_rows: List[Dict[str, Any]] = []
    transferred_original_images = set()

    def ensure_original_image(row: Dict[str, Any]) -> None:
        relative = row["image"]
        if relative in transferred_original_images:
            return
        source = resolve_dataset_image(input_dir, relative)
        destination = output_dir / Path(*PurePosixPath(relative).parts)
        transfer_file(source, destination, args.transfer, args.overwrite)
        transferred_original_images.add(relative)

    if args.include_original:
        for row in rows:
            ensure_original_image(row)
            output_rows.append({key: row[key] for key in REQUIRED_DATA_KEYS})

    augmented_count = 0
    fallback_count = 0
    agent_fallback_count = 0
    solver_fallback_count = 0
    image_fallback_count = 0
    for source_id, agent_record in sorted(
        agent_failed.items(), key=lambda item: int(item[1]["source_index"])
    ):
        source_row = rows_by_id.get(source_id)
        if source_row is None:
            if args.allow_partial:
                LOGGER.warning("Skipping removed failed-Agent source id %s", source_id)
                continue
            raise RuntimeError(
                f"Failed Agent manifest source id does not exist in input dataset: {source_id}"
            )
        ensure_original_image(source_row)
        if not args.include_original:
            output_rows.append({key: source_row[key] for key in REQUIRED_DATA_KEYS})
        fallback_count += 1
        agent_fallback_count += 1

    for source_id, agent_record in sorted(
        agent_success.items(), key=lambda item: int(item[1]["source_index"])
    ):
        source_row = rows_by_id.get(source_id)
        if source_row is None:
            if args.allow_partial:
                LOGGER.warning("Skipping removed source id %s", source_id)
                continue
            raise RuntimeError(f"Agent manifest source id does not exist in input dataset: {source_id}")

        previous_answer = str(source_row["answer"])
        recorded_agent_answer = agent_record.get("previous_answer")
        if (
            recorded_agent_answer is not None
            and str(recorded_agent_answer) != previous_answer
        ):
            raise RuntimeError(
                f"Agent answer lineage mismatch for source id {source_id}: the manifest "
                "was generated from a different previous-round answer. Reset this shard."
            )

        augmented_id = build_augmented_id(source_id, args.mode, args.round_name)
        if args.mode == MODE_TEXT:
            solver_record = solver_success.get(source_id)
            if solver_record is None:
                continue
            recorded_solver_answer = solver_record.get("previous_answer")
            if (
                recorded_solver_answer is not None
                and str(recorded_solver_answer) != previous_answer
            ):
                raise RuntimeError(
                    f"TTRL answer lineage mismatch for source id {source_id}: the solver "
                    "manifest was generated from a different previous-round answer. "
                    "Reset the solver shard."
                )
            ensure_original_image(source_row)
            if solver_record.get("fallback_to_source") is True:
                fallback_count += 1
                solver_fallback_count += 1
                if not args.include_original:
                    output_rows.append({key: source_row[key] for key in REQUIRED_DATA_KEYS})
                continue
            # Text editing changes the problem. Its label must therefore be the
            # TTRL consensus for that new problem, never the previous answer or
            # an answer proposed by the question-generation Agent.
            ttrl_answer = str(solver_record["answer"])
            augmented_row = {
                "id": augmented_id,
                "image": source_row["image"],
                "problem": str(agent_record["problem"]),
                "answer": ttrl_answer,
            }
        else:
            image_record = image_success.get(source_id)
            if image_record is None:
                ensure_original_image(source_row)
                if not args.include_original:
                    output_rows.append({key: source_row[key] for key in REQUIRED_DATA_KEYS})
                fallback_count += 1
                image_fallback_count += 1
                continue
            edited_source = work_dir / Path(*PurePosixPath(str(image_record["edited_file"])).parts)
            if not edited_source.is_file():
                LOGGER.warning(
                    "Edited image is missing for source id %s; preserving the source row: %s",
                    source_id,
                    edited_source,
                )
                ensure_original_image(source_row)
                if not args.include_original:
                    output_rows.append({key: source_row[key] for key in REQUIRED_DATA_KEYS})
                fallback_count += 1
                image_fallback_count += 1
                continue
            image_relative = PurePosixPath("images", f"aug_{stable_token(source_id, args.mode, args.round_name)}.png").as_posix()
            edited_destination = output_dir / Path(*PurePosixPath(image_relative).parts)
            transfer_file(edited_source, edited_destination, args.transfer, args.overwrite)
            # Image editing is constrained to be semantics preserving, so both
            # the problem and label are carried over from the previous round.
            augmented_row = {
                "id": augmented_id,
                "image": image_relative,
                "problem": source_row["problem"],
                "answer": previous_answer,
            }
        output_rows.append(augmented_row)
        augmented_count += 1

    if not output_rows:
        raise RuntimeError("Finalized dataset would be empty.")
    output_ids = [row["id"] for row in output_rows]
    if len(output_ids) != len(set(output_ids)):
        raise RuntimeError("Finalized dataset contains duplicate ids.")
    if not args.include_original and not args.allow_partial and len(output_rows) != expected:
        raise RuntimeError(
            f"{args.mode} finalization produced {len(output_rows)} rows but expected {expected}; "
            "refusing to change the per-round training size."
        )

    write_json_atomic(output_data_path, output_rows)
    summary = {
        "mode": args.mode,
        "round_name": args.round_name,
        "source_dataset": str(input_dir),
        "work_dir": str(work_dir),
        "include_original": args.include_original,
        "source_rows": len(rows),
        "agent_expected": expected,
        "agent_success": len(agent_success),
        "agent_failed": len(agent_failed),
        "solver_success": len(solver_success) if args.mode == MODE_TEXT else None,
        "augmented_rows": augmented_count,
        "fallback_rows": fallback_count,
        "agent_fallback_rows": agent_fallback_count,
        "solver_fallback_rows": solver_fallback_count,
        "image_fallback_rows": image_fallback_count,
        "total_rows": len(output_rows),
        "answer_policy": (
            "ttrl_consensus_for_text_problem; fallback_preserves_previous_row"
            if args.mode == MODE_TEXT
            else "carry_previous_round_answer; fallback_preserves_previous_row"
        ),
        "transfer": args.transfer,
    }
    write_json_atomic(output_dir / "augmentation_meta.json", summary)
    LOGGER.info(
        "Finalized %s: originals=%d, augmented=%d, fallback=%d, total=%d",
        output_dir,
        len(rows) if args.include_original else 0,
        augmented_count,
        fallback_count,
        len(output_rows),
    )


def merge_datasets(args: argparse.Namespace) -> None:
    input_dirs = [Path(value).expanduser().resolve() for value in args.input_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if len(set(input_dirs)) != len(input_dirs):
        raise ValueError("--input-dirs contains the same dataset more than once.")
    if output_dir in input_dirs:
        raise ValueError("The merge output directory must differ from every input dataset.")

    output_data_path = output_dir / "data.json"
    if output_data_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_data_path} already exists. Pass --overwrite to rebuild it.")

    merged_rows: List[Dict[str, Any]] = []
    seen_ids = set()
    destination_sources: Dict[str, Path] = {}
    input_counts: Dict[str, int] = {}
    renamed_collisions = 0

    for dataset_index, input_dir in enumerate(input_dirs):
        rows = load_dataset_rows(input_dir)
        input_counts[str(input_dir)] = len(rows)
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id across merged datasets: {row['id']}")
            seen_ids.add(row["id"])

            source_image = resolve_dataset_image(input_dir, row["image"])
            output_relative = row["image"]
            previous_source = destination_sources.get(output_relative)
            if previous_source is not None and not files_identical(previous_source, source_image):
                original_name = PurePosixPath(output_relative).name
                collision_key = f"{dataset_index}:{row['id']}:{output_relative}:{file_sha256(source_image)}"
                collision_token = hashlib.sha256(collision_key.encode("utf-8")).hexdigest()[:16]
                output_relative = PurePosixPath("images", f"merge_{collision_token}_{original_name}").as_posix()
                while output_relative in destination_sources and not files_identical(
                    destination_sources[output_relative], source_image
                ):
                    collision_token = hashlib.sha256((collision_token + row["id"]).encode("utf-8")).hexdigest()[:16]
                    output_relative = PurePosixPath(
                        "images", f"merge_{collision_token}_{original_name}"
                    ).as_posix()
                renamed_collisions += 1

            destination_sources.setdefault(output_relative, source_image)
            destination_image = output_dir / Path(*PurePosixPath(output_relative).parts)
            transfer_file(source_image, destination_image, args.transfer, args.overwrite)
            merged_rows.append(
                {
                    "id": row["id"],
                    "image": output_relative,
                    "problem": row["problem"],
                    "answer": row["answer"],
                }
            )

    if not merged_rows:
        raise RuntimeError("Merged dataset would be empty.")
    write_json_atomic(output_data_path, merged_rows)
    write_json_atomic(
        output_dir / "merge_meta.json",
        {
            "round_name": args.round_name,
            "input_datasets": input_counts,
            "total_rows": len(merged_rows),
            "unique_images": len(destination_sources),
            "renamed_image_collisions": renamed_collisions,
            "transfer": args.transfer,
        },
    )
    LOGGER.info(
        "Merged %d datasets into %s: rows=%d, unique_images=%d, renamed_collisions=%d",
        len(input_dirs),
        output_dir,
        len(merged_rows),
        len(destination_sources),
        renamed_collisions,
    )


def split_rows_for_augmentation(
    rows: Sequence[Dict[str, Any]], image_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < image_ratio < 1.0:
        raise ValueError("--image-ratio must be strictly between 0 and 1.")
    image_count = int(len(rows) * image_ratio + 0.5)
    if rows:
        image_count = min(max(image_count, 1), len(rows) - 1)

    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest(),
            row["id"],
        ),
    )
    image_ids = {row["id"] for row in ranked[:image_count]}
    text_rows = [row for row in rows if row["id"] not in image_ids]
    image_rows = [row for row in rows if row["id"] in image_ids]
    return text_rows, image_rows


def partition_dataset(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    text_output_dir = Path(args.text_output_dir).expanduser().resolve()
    image_output_dir = Path(args.image_output_dir).expanduser().resolve()
    output_dirs = (text_output_dir, image_output_dir)
    if text_output_dir == image_output_dir:
        raise ValueError("Text and image partition output directories must differ.")
    if text_output_dir in image_output_dir.parents or image_output_dir in text_output_dir.parents:
        raise ValueError("Partition output directories must not contain one another.")
    for output_dir in output_dirs:
        if (
            output_dir == input_dir
            or input_dir in output_dir.parents
            or output_dir in input_dir.parents
        ):
            raise ValueError("Partition outputs and the source dataset must not contain one another.")
        data_path = output_dir / "data.json"
        if data_path.exists() and not args.overwrite:
            raise FileExistsError(f"{data_path} already exists. Pass --overwrite to rebuild it.")

    rows = load_dataset_rows(input_dir)
    if len(rows) < 2:
        raise ValueError("At least two source rows are required for text/image partitioning.")
    text_rows, image_rows = split_rows_for_augmentation(rows, args.image_ratio, args.seed)
    partitions = (
        ("text", text_output_dir, text_rows),
        (MODE_IMAGE_EDIT, image_output_dir, image_rows),
    )
    for partition_name, output_dir, partition_rows in partitions:
        transferred_images = set()
        for row in partition_rows:
            relative = row["image"]
            if relative in transferred_images:
                continue
            source_image = resolve_dataset_image(input_dir, relative)
            destination_image = output_dir / Path(*PurePosixPath(relative).parts)
            transfer_file(source_image, destination_image, args.transfer, args.overwrite)
            transferred_images.add(relative)

        write_json_atomic(
            output_dir / "data.json",
            [{key: row[key] for key in REQUIRED_DATA_KEYS} for row in partition_rows],
        )
        write_json_atomic(
            output_dir / "partition_meta.json",
            {
                "partition": partition_name,
                "source_dataset": str(input_dir),
                "source_rows": len(rows),
                "partition_rows": len(partition_rows),
                "image_ratio": args.image_ratio,
                "seed": args.seed,
                "transfer": args.transfer,
            },
        )

    if len(text_rows) + len(image_rows) != len(rows):
        raise RuntimeError("Partition row counts do not cover the source dataset exactly.")
    if {row["id"] for row in text_rows} & {row["id"] for row in image_rows}:
        raise RuntimeError("Text and image partitions overlap.")
    LOGGER.info(
        "Partitioned %s: text=%d (%.2f%%), image-edit=%d (%.2f%%)",
        input_dir,
        len(text_rows),
        100.0 * len(text_rows) / len(rows),
        len(image_rows),
        100.0 * len(image_rows) / len(rows),
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", required=True, help="Source dataset directory containing data.json and images/.")
    parser.add_argument("--work-dir", required=True, help="Separate directory for resumable manifests and edited images.")


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--reset-shard",
        action="store_true",
        help="Discard this stage's manifest for the selected shard before starting.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Run the Qwen2.5-VL augmentation agent.")
    add_dataset_args(generate_parser)
    add_shard_args(generate_parser)
    generate_parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    generate_parser.add_argument(
        "--agent-model",
        required=True,
        help="Original Qwen2.5-VL-7B model id/path or a merged trained actor checkpoint.",
    )
    generate_parser.add_argument(
        "--processor-model",
        default=None,
        help="Optional original processor path when a trained checkpoint does not contain processor files.",
    )
    generate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    generate_parser.add_argument("--max-model-len", type=int, default=16384)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    generate_parser.add_argument("--batch-size", type=int, default=32)
    generate_parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    generate_parser.add_argument("--max-tokens", type=int, default=384)
    generate_parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=1,
        help="Retry only invalid Agent outputs with a different sampling seed.",
    )
    generate_parser.add_argument(
        "--min-pixels",
        type=int,
        default=256 * 28 * 28,
        help="Minimum Qwen visual pixels (default: 256 visual tokens).",
    )
    generate_parser.add_argument(
        "--max-pixels",
        type=int,
        default=2048 * 28 * 28,
        help="Maximum Qwen visual pixels (default: 2048 visual tokens).",
    )
    generate_parser.add_argument("--temperature", type=float, default=0.4)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument("--seed", type=int, default=1)
    generate_parser.add_argument("--max-samples", type=int, default=None)
    generate_parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    generate_parser.set_defaults(handler=generate_agent_outputs)

    solver_parser = subparsers.add_parser(
        "solve-text",
        help="Solve generated text augmentations with multi-round Qwen2.5-VL TTRL voting.",
    )
    add_dataset_args(solver_parser)
    add_shard_args(solver_parser)
    solver_parser.add_argument(
        "--solver-model",
        required=True,
        help="Original or trained Qwen2.5-VL-7B answer model.",
    )
    solver_parser.add_argument(
        "--processor-model",
        default=None,
        help="Optional original processor path when the solver checkpoint has no processor files.",
    )
    solver_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    solver_parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16"),
        default="auto",
    )
    solver_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    solver_parser.add_argument("--batch-size", type=int, default=51)
    solver_parser.add_argument("--max-model-len", type=int, default=8192)
    solver_parser.add_argument("--max-num-seqs", type=int, default=256)
    solver_parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    solver_parser.add_argument("--max-tokens", type=int, default=1024)
    solver_parser.add_argument("--votes-per-round", type=int, default=5)
    solver_parser.add_argument("--min-vote-rounds", type=int, default=1)
    solver_parser.add_argument("--max-vote-rounds", type=int, default=1)
    solver_parser.add_argument("--min-valid-votes", type=int, default=2)
    solver_parser.add_argument("--min-agree-votes", type=int, default=2)
    solver_parser.add_argument("--consensus-threshold", type=float, default=0.4)
    solver_parser.add_argument("--temperature", type=float, default=0.7)
    solver_parser.add_argument("--top-p", type=float, default=0.95)
    solver_parser.add_argument("--seed", type=int, default=17)
    solver_parser.add_argument("--store-raw-outputs", action="store_true")
    solver_parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    solver_parser.set_defaults(handler=solve_text_answers)

    edit_parser = subparsers.add_parser("edit-images", help="Materialize image edits with Qwen-Image-Edit-2511.")
    add_dataset_args(edit_parser)
    add_shard_args(edit_parser)
    edit_parser.add_argument("--image-model", default="Qwen/Qwen-Image-Edit-2511")
    edit_parser.add_argument("--round-name", default="round1")
    edit_parser.add_argument("--device", default="cuda")
    edit_parser.add_argument("--cpu-offload", action="store_true")
    edit_parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    edit_parser.add_argument("--guidance-scale", type=float, default=1.0)
    edit_parser.add_argument("--num-inference-steps", type=int, default=20)
    edit_parser.add_argument("--negative-prompt", default=" ")
    edit_parser.add_argument("--seed", type=int, default=1)
    edit_parser.add_argument(
        "--preserve-size",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    edit_parser.set_defaults(handler=edit_images)

    finalize_parser = subparsers.add_parser("finalize", help="Merge successful augmentations into a new dataset.")
    add_dataset_args(finalize_parser)
    finalize_parser.add_argument("--output-dir", required=True)
    finalize_parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    finalize_parser.add_argument("--round-name", default="round1")
    finalize_parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include original rows as well as augmented rows (default: false).",
    )
    finalize_parser.add_argument("--allow-partial", action="store_true")
    finalize_parser.add_argument("--overwrite", action="store_true")
    finalize_parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="hardlink")
    finalize_parser.set_defaults(handler=finalize_dataset)

    merge_parser = subparsers.add_parser(
        "merge-datasets",
        help="Merge cumulative and newly augmented datasets without duplicating identical image files.",
    )
    merge_parser.add_argument("--input-dirs", nargs="+", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.add_argument("--round-name", default="round1")
    merge_parser.add_argument("--overwrite", action="store_true")
    merge_parser.add_argument("--transfer", choices=("copy", "hardlink", "symlink"), default="hardlink")
    merge_parser.set_defaults(handler=merge_datasets)

    partition_parser = subparsers.add_parser(
        "partition-dataset",
        help="Deterministically split one round into disjoint text and image-edit subsets.",
    )
    partition_parser.add_argument("--input-dir", required=True)
    partition_parser.add_argument("--text-output-dir", required=True)
    partition_parser.add_argument("--image-output-dir", required=True)
    partition_parser.add_argument("--image-ratio", type=float, default=0.1)
    partition_parser.add_argument("--seed", type=int, default=2025)
    partition_parser.add_argument("--overwrite", action="store_true")
    partition_parser.add_argument(
        "--transfer",
        choices=("copy", "hardlink", "symlink"),
        default="hardlink",
    )
    partition_parser.set_defaults(handler=partition_dataset)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    args.handler(args)


if __name__ == "__main__":
    main()
