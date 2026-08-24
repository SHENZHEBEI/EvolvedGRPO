"""Batch reward client for iterative question-generator GRPO training.

The GPU-heavy evaluator is deliberately kept outside the EasyQ1 trainer.  It
returns auditable evidence (TTRL votes, CLIP similarity, and critic votes); this
module owns the exact reward formula so a remote scorer cannot silently change
the optimization objective.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from data_augmentation.augment_dataset import (
    MODE_IMAGE_EDIT,
    MODE_TEXT,
    SUPPORTED_MODES,
    normalize_agent_payload,
)


DEFAULT_VOTES = 5
DEFAULT_PROBABILITY_FLOOR = 1.0 / DEFAULT_VOTES
DEFAULT_MIN_TEXT_AGREEMENT = 2
DEFAULT_MIN_CLIP_SIMILARITY = DEFAULT_PROBABILITY_FLOOR


def empty_score() -> Dict[str, float]:
    """Return a metric-complete zero score for an invalid candidate."""

    return {
        "overall": 0.0,
        "valid": 0.0,
        "format": 0.0,
        "step_probability": 0.0,
        "answer_probability": 0.0,
        "reliability_log": 0.0,
        "difficulty_log": 0.0,
    }


def parse_context(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("ground_truth is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("ground_truth must be a JSON object")

    required = ("source_id", "mode", "previous_problem", "previous_answer", "image_path")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"ground_truth is missing fields: {missing}")
    if value["mode"] not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {value['mode']!r}")
    if not str(value["previous_problem"]).strip():
        raise ValueError("previous_problem is empty")
    if not str(value["previous_answer"]).strip():
        raise ValueError("previous_answer is empty")
    if not str(value["image_path"]).strip():
        raise ValueError("image_path is empty")
    return dict(value)


def normalize_candidate(predict_str: str, context: Mapping[str, Any]) -> Dict[str, str]:
    """Strictly parse the JSON schema requested by the generator prompt.

    Offline augmentation accepts plain-text follow-ups as a recovery policy.
    GRPO must not do that: malformed JSON is precisely what the format gate is
    meant to train away.
    """

    try:
        payload = json.loads(str(predict_str).strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Policy response must be exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("Policy response must be a JSON object")
    mode = str(context["mode"])
    required_key = "edit_instruction" if mode == MODE_IMAGE_EDIT else "follow_up_problem"
    if set(payload) != {required_key}:
        raise ValueError(f"Expected exactly one JSON field: {required_key}")
    return normalize_agent_payload(
        mode,
        json.dumps(payload, ensure_ascii=False),
        str(context["previous_problem"]),
    )


def candidate_key(context: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    payload = {
        "source_id": str(context["source_id"]),
        "mode": str(context["mode"]),
        "previous_problem": str(context["previous_problem"]),
        "previous_answer": str(context["previous_answer"]),
        "image_path": str(context["image_path"]),
        "round_name": str(context.get("round_name", "")),
        "candidate": dict(candidate),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1], got {result!r}")
    return result


def compute_reward_from_evidence(
    mode: str,
    evidence: Mapping[str, Any],
    *,
    votes: int = DEFAULT_VOTES,
    probability_floor: float = DEFAULT_PROBABILITY_FLOOR,
    min_text_agreement: int = DEFAULT_MIN_TEXT_AGREEMENT,
    min_clip_similarity: float = DEFAULT_MIN_CLIP_SIMILARITY,
) -> Dict[str, float]:
    """Compute the simplified paper reward from model-independent evidence.

    ``format = log(votes)`` is the requested valid-output floor.  With
    ``probability_floor = 1 / votes`` and a valid reliability probability of at
    least that floor, the total is never negative.  Invalid question/answer
    chains are hard-gated to zero.
    """

    if votes <= 0:
        raise ValueError("votes must be positive")
    if not 0.0 < probability_floor <= 1.0:
        raise ValueError("probability_floor must be in (0, 1]")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {mode!r}")
    if not bool(evidence.get("question_valid")) or not bool(evidence.get("answer_valid")):
        return empty_score()

    answer_probability = _probability(evidence.get("answer_probability"), "answer_probability")
    if mode == MODE_TEXT:
        total_votes = int(evidence.get("ttrl_total_votes", votes))
        winning_votes = int(evidence.get("ttrl_winning_votes", 0))
        if total_votes != votes or winning_votes < min_text_agreement or winning_votes > total_votes:
            return empty_score()
        step_probability = winning_votes / total_votes
    else:
        step_probability = _probability(evidence.get("clip_similarity"), "clip_similarity")
        if step_probability < min_clip_similarity:
            return empty_score()

    # The valid gate above guarantees p_step >= 1/votes under the default
    # protocol.  Keep an explicit guard for changed/custom configurations.
    if step_probability < probability_floor:
        return empty_score()

    reliability_log = math.log(step_probability)
    difficulty_log = -math.log(max(probability_floor, answer_probability))
    format_reward = math.log(float(votes))
    overall = reliability_log + difficulty_log + format_reward
    # Avoid a tiny negative number from floating point round-off at the exact
    # lower bound (p_step=1/5, p_answer=1).
    overall = max(0.0, overall)
    return {
        "overall": overall,
        "valid": 1.0,
        "format": format_reward,
        "step_probability": step_probability,
        "answer_probability": answer_probability,
        "reliability_log": reliability_log,
        "difficulty_log": difficulty_log,
    }


def _post_json(endpoint: str, payload: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    url = endpoint.rstrip("/") + "/score"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # HTTPError is also a URLError, so handle it first and preserve the
        # server's JSON error.  Without this, a useful CUDA/vLLM traceback is
        # reduced to the opaque message "HTTP Error 400: Bad Request".
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        detail = error_body[:2000].strip()
        if detail:
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict) and parsed.get("error"):
                    detail = str(parsed["error"])
            except json.JSONDecodeError:
                pass
        suffix = f"; server error: {detail}" if detail else ""
        raise RuntimeError(
            f"Question reward service request failed at {url}: "
            f"HTTP {exc.code} {exc.reason}{suffix}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Question reward service request failed at {url}: {exc}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Question reward service returned invalid JSON: {body[:500]!r}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Question reward service response must be a JSON object")
    return value


def score_batch(
    predict_strs: Sequence[str],
    ground_truths: Sequence[str],
    *,
    endpoint: Optional[str] = None,
    timeout: float = 7200.0,
    votes: int = DEFAULT_VOTES,
    probability_floor: float = DEFAULT_PROBABILITY_FLOOR,
    min_text_agreement: int = DEFAULT_MIN_TEXT_AGREEMENT,
    min_clip_similarity: float = DEFAULT_MIN_CLIP_SIMILARITY,
) -> List[Dict[str, float]]:
    """EasyQ1 batch score entry point.

    The service is called only for locally well-formed candidates.  Every
    malformed response receives an immediate zero without consuming TTRL,
    image-edit, CLIP, or critic inference.
    """

    if len(predict_strs) != len(ground_truths):
        raise ValueError("predict_strs and ground_truths must have equal length")
    endpoint = (
        endpoint
        or os.environ.get("QUESTION_REWARD_ENDPOINTS")
        or os.environ.get("QUESTION_REWARD_ENDPOINT")
    )
    if not endpoint:
        raise RuntimeError(
            "QUESTION_REWARD_ENDPOINT(S) is unset. Start the GPU reward service and pass "
            "worker.reward.score_function_kwargs.endpoint, or export the variable."
        )
    endpoints = [value.strip() for value in endpoint.split(",") if value.strip()]
    if not endpoints:
        raise RuntimeError("No usable question reward endpoint was provided")

    scores = [empty_score() for _ in predict_strs]
    requests_by_key: Dict[str, Dict[str, Any]] = {}
    output_indices_by_key: Dict[str, List[int]] = {}
    for index, (predict_str, ground_truth) in enumerate(zip(predict_strs, ground_truths)):
        try:
            context = parse_context(ground_truth)
            candidate = normalize_candidate(predict_str, context)
        except (TypeError, ValueError, KeyError):
            continue
        key = candidate_key(context, candidate)
        requests_by_key.setdefault(
            key,
            {"candidate_key": key, "context": context, "candidate": candidate},
        )
        output_indices_by_key.setdefault(key, []).append(index)

    if not requests_by_key:
        return scores

    # Stable hash routing keeps a candidate on the same GPU shard across
    # retries, maximizing evidence/image cache reuse.  Duplicate rollouts are
    # sent only once and receive the same evidence afterward.
    endpoint_requests: List[List[Dict[str, Any]]] = [[] for _ in endpoints]
    for key, request_item in requests_by_key.items():
        shard = int(key[:16], 16) % len(endpoints)
        endpoint_requests[shard].append(request_item)

    def call_shard(shard: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        shard_requests = endpoint_requests[shard]
        response = _post_json(
            endpoints[shard],
            {"protocol_version": 1, "votes": votes, "candidates": shard_requests},
            timeout,
        )
        return shard_requests, response

    shard_results: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    active_shards = [index for index, values in enumerate(endpoint_requests) if values]
    with ThreadPoolExecutor(max_workers=len(active_shards)) as executor:
        futures = [executor.submit(call_shard, shard) for shard in active_shards]
        for future in futures:
            shard_results.append(future.result())

    for shard_requests, response in shard_results:
        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(shard_requests):
            raise RuntimeError(
                "Question reward service returned a result count that does not match the request"
            )
        for request_item, evidence in zip(shard_requests, results):
            if not isinstance(evidence, dict):
                raise RuntimeError("Question reward evidence must be a JSON object")
            key = request_item["candidate_key"]
            if evidence.get("candidate_key") != key:
                raise RuntimeError("Question reward service candidate_key mismatch")
            score = compute_reward_from_evidence(
                str(request_item["context"]["mode"]),
                evidence,
                votes=votes,
                probability_floor=probability_floor,
                min_text_agreement=min_text_agreement,
                min_clip_similarity=min_clip_similarity,
            )
            for output_index in output_indices_by_key[key]:
                scores[output_index] = dict(score)
    return scores


main_batch = score_batch
