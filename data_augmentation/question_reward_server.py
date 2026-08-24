"""GPU reward service for iterative question-generator GRPO.

The service performs the expensive, candidate-dependent work:

* text: five privileged TTRL solves from (previous answer, new problem);
* image: Qwen-Image-Edit plus CLIP cosine similarity;
* both: five unprivileged multimodal solves by the configured answer model.

It returns evidence only.  The EasyQ1-side reward client computes the final
formula and applies the hard zero gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from data_augmentation.augment_dataset import (
    MODE_IMAGE_EDIT,
    MODE_TEXT,
    SAFE_EDIT_SUFFIX,
    TEXT_SOLVER_PROMPT,
    answers_equivalent,
    build_text_vllm_input,
    build_vllm_input,
    extract_boxed_answer,
    load_qwen_image_edit_pipeline_compat,
    summarize_votes,
)


LOGGER = logging.getLogger("question_reward_server")

CRITIC_SOLVER_PROMPT = r"""Solve the mathematical problem using the supplied image.
Do not use any hidden previous answer. Show concise reasoning and end with exactly one
final answer in \boxed{{}}. Do not output multiple alternative answers.

Problem:
{problem}
"""


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
        ) as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _error_evidence(candidate_key: str, error: str) -> Dict[str, Any]:
    return {
        "candidate_key": candidate_key,
        "question_valid": False,
        "answer_valid": False,
        "answer_probability": 0.0,
        "error": error,
    }


class QuestionRewardEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from mathruler.grader import extract_boxed_content, grade_answer
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "Reward service requires vllm, transformers, qwen-vl-utils, and mathruler. "
                f"Activate augmentation_env.sh first: {exc!r}"
            ) from exc

        self.args = args
        base_cache_dir = Path(args.cache_dir).expanduser().resolve()
        evaluator_signature = {
            "protocol_version": 1,
            "answer_model": str(args.answer_model),
            "processor_model": str(args.processor_model or args.answer_model),
            "image_model": str(args.image_model),
            "clip_model": str(args.clip_model),
            "votes": args.votes,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "min_agree_votes": args.min_agree_votes,
            "consensus_threshold": args.consensus_threshold,
            "image_num_inference_steps": args.image_num_inference_steps,
            "image_true_cfg_scale": args.image_true_cfg_scale,
            "image_guidance_scale": args.image_guidance_scale,
            "safe_edit_suffix_sha256": hashlib.sha256(
                SAFE_EDIT_SUFFIX.encode("utf-8")
            ).hexdigest(),
            "seed": args.seed,
        }
        serialized_signature = json.dumps(
            evaluator_signature, sort_keys=True, separators=(",", ":")
        )
        namespace = hashlib.sha256(serialized_signature.encode("utf-8")).hexdigest()[:20]
        self.cache_dir = base_cache_dir / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.cache_dir / "evaluator.json", evaluator_signature)
        LOGGER.info("Reward evidence cache namespace: %s", self.cache_dir)
        self.image_dir = self.cache_dir / "edited_images"
        self.extract_boxed_content = extract_boxed_content
        self.grade_answer = grade_answer
        self.process_vision_info = process_vision_info
        self.SamplingParams = SamplingParams
        self.processor = AutoProcessor.from_pretrained(
            args.processor_model or args.answer_model,
            trust_remote_code=args.trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("Answer-model processor has no tokenizer for length checks.")
        # The same configured answer model is intentionally used for the
        # privileged TTRL path and the unprivileged difficulty critic.  In the
        # coevolution launcher this is A_k while Q_k is being trained.
        self.llm = LLM(
            model=args.answer_model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            limit_mm_per_prompt={"image": 1, "video": 0},
            enable_sleep_mode=True,
        )
        self._answer_model_awake = True
        self._image_pipeline = None
        self._clip_model = None
        self._clip_processor = None
        # The service is started before EasyQ1 initializes its actor workers on
        # the same eight A800 GPUs. Release every vLLM allocation while idle so Ray
        # can initialize and use the full cards.
        self._sleep_answer_model()

    def _wake_answer_model(self) -> None:
        if not self._answer_model_awake:
            self.llm.wake_up()
            self._answer_model_awake = True

    def _sleep_answer_model(self) -> None:
        if self._answer_model_awake:
            self.llm.sleep(level=1)
            self._answer_model_awake = False

    def _offload_clip(self) -> None:
        if self._clip_model is not None:
            self._clip_model.to("cpu")

    def _offload_image_pipeline(self) -> None:
        if self._image_pipeline is None:
            return
        pipeline = self._image_pipeline
        if self.args.unload_image_after_request:
            self._image_pipeline = None
            del pipeline
            gc.collect()
        elif not self.args.image_cpu_offload:
            pipeline.to("cpu")

    def _release_all_gpu_memory(self) -> None:
        import torch

        # Keep only CPU-side lightweight state between HTTP requests.  The
        # trainer resumes old-logprob/ref/actor work only after the response is
        # returned, so this release is part of the synchronization contract.
        self._offload_image_pipeline()
        self._offload_clip()
        self._sleep_answer_model()
        gc.collect()
        torch.cuda.empty_cache()

    def _cache_path(self, candidate_key: str) -> Path:
        return self.cache_dir / "evidence" / candidate_key[:2] / f"{candidate_key}.json"

    def _load_cache(self, candidate_key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(candidate_key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(value, dict) and value.get("candidate_key") == candidate_key:
            return value
        return None

    def _save_cache(self, value: Mapping[str, Any]) -> None:
        _write_json_atomic(self._cache_path(str(value["candidate_key"])), value)

    def _extract_answers(self, outputs: Any) -> List[str]:
        answers: List[str] = []
        for candidate in getattr(outputs, "outputs", ()):
            answer = extract_boxed_answer(candidate.text, self.extract_boxed_content)
            if answer is not None:
                answers.append(answer)
        return answers

    def _generate_in_safe_batches(
        self,
        model_inputs: Sequence[Dict[str, Any]],
        params: Any,
        *,
        samples_per_prompt: int,
        stage: str,
    ) -> List[Any]:
        """Keep each vLLM submission within max_num_seqs.

        A reward shard receives roughly 320 unique policy candidates in the
        current 512-prompt x 5-rollout job.  Each reward prompt itself requests
        five samples, so submitting every prompt in one call can temporarily
        create far more sequences than the per-shard scheduler limit.  vLLM can
        queue requests internally, but bounding each call also bounds Python
        output state and makes failures attributable to one small batch.
        """

        if samples_per_prompt <= 0:
            raise ValueError("samples_per_prompt must be positive")
        prompt_batch_size = max(1, self.args.max_num_seqs // samples_per_prompt)
        generated: List[Any] = []
        LOGGER.info(
            "%s generation: prompts=%d, samples_per_prompt=%d, prompt_batch_size=%d",
            stage,
            len(model_inputs),
            samples_per_prompt,
            prompt_batch_size,
        )
        for start in range(0, len(model_inputs), prompt_batch_size):
            batch = model_inputs[start : start + prompt_batch_size]
            generated.extend(self.llm.generate(batch, params, use_tqdm=False))
        if len(generated) != len(model_inputs):
            raise RuntimeError(
                f"{stage} generation returned {len(generated)} outputs for "
                f"{len(model_inputs)} prompts"
            )
        return generated

    def _load_image_pipeline(self) -> Any:
        if self._image_pipeline is not None:
            return self._image_pipeline
        try:
            import torch
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as exc:
            raise RuntimeError(f"Image reward requires QwenImageEditPlusPipeline: {exc!r}") from exc
        pipeline = load_qwen_image_edit_pipeline_compat(
            QwenImageEditPlusPipeline,
            self.args.image_model,
            torch.bfloat16,
        )
        if self.args.image_cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.args.image_device)
        pipeline.set_progress_bar_config(disable=True)
        self._image_pipeline = pipeline
        return pipeline

    def _load_clip(self) -> Tuple[Any, Any]:
        if self._clip_model is not None:
            self._clip_model.to(self.args.clip_device)
            return self._clip_model, self._clip_processor
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(f"CLIP reward requires transformers and torch: {exc!r}") from exc
        processor = CLIPProcessor.from_pretrained(self.args.clip_model)
        model = CLIPModel.from_pretrained(
            self.args.clip_model,
            torch_dtype=torch.float16 if str(self.args.clip_device).startswith("cuda") else torch.float32,
        )
        model.to(self.args.clip_device)
        model.eval()
        self._clip_model = model
        self._clip_processor = processor
        return model, processor

    def _edit_and_clip(
        self, candidate_key: str, source_image_path: Path, instruction: str
    ) -> Tuple[Path, float]:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        pipeline = self._load_image_pipeline()
        edited_path = self.image_dir / f"{candidate_key}.png"
        with Image.open(source_image_path) as opened:
            source_image = opened.convert("RGB")
            source_size = source_image.size

        if not edited_path.is_file():
            seed = self.args.seed + int(candidate_key[:8], 16)
            generator_device = self.args.image_device if str(self.args.image_device).startswith("cuda") else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            with torch.inference_mode():
                result = pipeline(
                    image=[source_image],
                    prompt=instruction.strip() + SAFE_EDIT_SUFFIX,
                    generator=generator,
                    true_cfg_scale=self.args.image_true_cfg_scale,
                    guidance_scale=self.args.image_guidance_scale,
                    num_inference_steps=self.args.image_num_inference_steps,
                    num_images_per_prompt=1,
                )
            edited = result.images[0].convert("RGB")
            if edited.size != source_size:
                edited = edited.resize(source_size, Image.Resampling.LANCZOS)
            edited_path.parent.mkdir(parents=True, exist_ok=True)
            edited.save(edited_path, format="PNG")

        with Image.open(edited_path) as opened:
            edited_image = opened.convert("RGB")
        clip_model, clip_processor = self._load_clip()
        inputs = clip_processor(images=[source_image, edited_image], return_tensors="pt")
        clip_dtype = next(clip_model.parameters()).dtype
        pixel_values = inputs["pixel_values"].to(
            device=self.args.clip_device, dtype=clip_dtype
        )
        with torch.inference_mode():
            embeddings = clip_model.get_image_features(pixel_values=pixel_values)
            embeddings = functional.normalize(embeddings.float(), dim=-1)
            similarity = float((embeddings[0] * embeddings[1]).sum().item())
        # Direct cosine similarity is the paper-style sim term.  Clamp only to
        # its mathematical probability domain; the client applies the 1/5
        # semantic-validity threshold before taking log.
        return edited_path, min(1.0, max(0.0, similarity))

    def _prepare_ttrl(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        context = item["context"]
        candidate = item["candidate"]
        prompt = TEXT_SOLVER_PROMPT.format(
            problem=candidate["problem"],
            answer=context["previous_answer"],
        )
        model_input = build_text_vllm_input(self.processor, prompt)
        prompt_tokens = len(
            self.tokenizer.encode(model_input["prompt"], add_special_tokens=False)
        )
        required = prompt_tokens + self.args.max_tokens
        if required > self.args.max_model_len:
            raise ValueError(
                f"TTRL needs {prompt_tokens} prompt + {self.args.max_tokens} output "
                f"tokens, exceeding max_model_len={self.args.max_model_len}"
            )
        return model_input

    def _prepare_critic(
        self, problem: str, image_path: Path
    ) -> Dict[str, Any]:
        model_input = build_vllm_input(
            self.processor,
            self.process_vision_info,
            image_path,
            CRITIC_SOLVER_PROMPT.format(problem=problem),
            min_pixels=self.args.min_pixels,
            max_pixels=self.args.max_pixels,
        )
        text_tokens = len(
            self.tokenizer.encode(model_input["prompt"], add_special_tokens=False)
        )
        maximum_visual_tokens = (self.args.max_pixels + 28 * 28 - 1) // (28 * 28)
        required = text_tokens + maximum_visual_tokens + self.args.max_tokens
        if required > self.args.max_model_len:
            raise ValueError(
                f"critic may need {text_tokens} text + {maximum_visual_tokens} visual + "
                f"{self.args.max_tokens} output tokens, exceeding "
                f"max_model_len={self.args.max_model_len}"
            )
        return model_input

    def evaluate(self, candidates: Sequence[Mapping[str, Any]], votes: int) -> List[Dict[str, Any]]:
        """Wake the reward model only for this synchronized reward request."""

        try:
            self._wake_answer_model()
            return self._evaluate_awake(candidates, votes)
        finally:
            # The HTTP response is not sent until this finishes.  Therefore the
            # EasyQ1 driver cannot start actor/ref computation while reward
            # allocations are still resident on the same GPUs.
            self._release_all_gpu_memory()

    def _evaluate_awake(
        self, candidates: Sequence[Mapping[str, Any]], votes: int
    ) -> List[Dict[str, Any]]:
        if votes != self.args.votes:
            raise ValueError(
                f"Request asks for {votes} votes, but server was started with {self.args.votes}."
            )
        results: List[Optional[Dict[str, Any]]] = [None] * len(candidates)
        pending: List[Tuple[int, Mapping[str, Any]]] = []
        for index, item in enumerate(candidates):
            candidate_key = str(item.get("candidate_key", ""))
            if len(candidate_key) != 64:
                results[index] = _error_evidence(candidate_key, "invalid candidate_key")
                continue
            cached = self._load_cache(candidate_key)
            if cached is not None:
                results[index] = cached
            else:
                pending.append((index, item))

        text_pending = [(index, item) for index, item in pending if item.get("context", {}).get("mode") == MODE_TEXT]
        image_pending = [(index, item) for index, item in pending if item.get("context", {}).get("mode") == MODE_IMAGE_EDIT]
        known_indices = {index for index, _ in text_pending + image_pending}
        for index, item in pending:
            if index not in known_indices:
                results[index] = _error_evidence(str(item.get("candidate_key", "")), "unsupported mode")

        # Stage 1A: privileged TTRL labels and p_step for text candidates.
        if text_pending:
            ttrl_inputs: List[Dict[str, Any]] = []
            ttrl_items: List[Tuple[int, Mapping[str, Any]]] = []
            for index, item in text_pending:
                try:
                    ttrl_inputs.append(self._prepare_ttrl(item))
                    ttrl_items.append((index, item))
                except Exception as exc:
                    results[index] = _error_evidence(str(item["candidate_key"]), f"TTRL input failed: {exc}")
            if ttrl_inputs:
                params = self.SamplingParams(
                    n=votes,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    seed=self.args.seed,
                )
                generated = self._generate_in_safe_batches(
                    ttrl_inputs,
                    params,
                    samples_per_prompt=votes,
                    stage="TTRL",
                )
                for (index, item), output in zip(ttrl_items, generated):
                    answers = self._extract_answers(output)
                    summary = summarize_votes(
                        answers, self.grade_answer, total_candidates=votes
                    )
                    resolved = (
                        summary["winning_votes"] >= self.args.min_agree_votes
                        and summary["candidate_share"] >= self.args.consensus_threshold
                        and summary["answer"] is not None
                    )
                    results[index] = {
                        "candidate_key": item["candidate_key"],
                        "question_valid": bool(resolved),
                        "answer_valid": bool(resolved),
                        "target_answer": summary["answer"] if resolved else None,
                        "ttrl_winning_votes": int(summary["winning_votes"]),
                        "ttrl_valid_votes": int(summary["valid_votes"]),
                        "ttrl_total_votes": votes,
                        "ttrl_vote_counts": summary["vote_counts"],
                        "answer_probability": 0.0,
                    }

        # Stage 1B: image materialization and CLIP p_step.  Qwen-Image-Edit is
        # much larger than the 7B answer model, so temporarily sleep vLLM and
        # offload the image pipeline before waking vLLM for the critic stage.
        if image_pending:
            import torch

            answer_model_slept = False
            try:
                if self.args.sleep_answer_during_image:
                    self._sleep_answer_model()
                    answer_model_slept = True
                # Do not retain a local reference to this very large pipeline.
                # _offload_image_pipeline() clears self._image_pipeline below;
                # a second local reference would otherwise keep all Qwen-Image
                # CUDA tensors alive while vLLM is being woken again.
                self._load_image_pipeline()
                if not self.args.image_cpu_offload:
                    self._image_pipeline.to(self.args.image_device)

                for index, item in image_pending:
                    try:
                        context = item["context"]
                        source_image = Path(str(context["image_path"])).expanduser().resolve()
                        if not source_image.is_file():
                            raise FileNotFoundError(f"source image not found: {source_image}")
                        edited_path, similarity = self._edit_and_clip(
                            str(item["candidate_key"]),
                            source_image,
                            str(item["candidate"]["edit_instruction"]),
                        )
                        target = str(context["previous_answer"]).strip()
                        results[index] = {
                            "candidate_key": item["candidate_key"],
                            "question_valid": True,
                            "answer_valid": bool(target),
                            "target_answer": target,
                            "clip_similarity": similarity,
                            "edited_image_path": str(edited_path),
                            "answer_probability": 0.0,
                        }
                    except Exception as exc:
                        LOGGER.exception("Image candidate failed: %s", item.get("candidate_key"))
                        results[index] = _error_evidence(
                            str(item["candidate_key"]), f"image/CLIP failed: {exc}"
                        )
            except Exception as exc:
                LOGGER.exception("Image reward stage initialization failed")
                for index, item in image_pending:
                    if results[index] is None:
                        results[index] = _error_evidence(
                            str(item["candidate_key"]),
                            f"image stage initialization failed: {exc}",
                        )
            finally:
                try:
                    self._offload_image_pipeline()
                    self._offload_clip()
                    torch.cuda.empty_cache()
                finally:
                    if answer_model_slept:
                        self._wake_answer_model()

        # Stage 2: difficulty.  The critic sees (new problem, current image),
        # never the previous answer.  Its empirical correct rate is /5 even if
        # some outputs omit a boxed answer.
        critic_inputs: List[Dict[str, Any]] = []
        critic_items: List[Tuple[int, Mapping[str, Any], str]] = []
        for index, item in pending:
            evidence = results[index]
            if not evidence or not evidence.get("question_valid") or not evidence.get("answer_valid"):
                continue
            try:
                context = item["context"]
                if context["mode"] == MODE_TEXT:
                    problem = str(item["candidate"]["problem"])
                    image_path = Path(str(context["image_path"])).expanduser().resolve()
                else:
                    problem = str(context["previous_problem"])
                    image_path = Path(str(evidence["edited_image_path"])).expanduser().resolve()
                critic_inputs.append(self._prepare_critic(problem, image_path))
                critic_items.append((index, item, str(evidence["target_answer"])))
            except Exception as exc:
                results[index] = _error_evidence(str(item["candidate_key"]), f"critic input failed: {exc}")

        if critic_inputs:
            params = self.SamplingParams(
                n=votes,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_tokens=self.args.max_tokens,
                seed=self.args.seed + 1,
            )
            generated = self._generate_in_safe_batches(
                critic_inputs,
                params,
                samples_per_prompt=votes,
                stage="critic",
            )
            for (index, item, target_answer), output in zip(critic_items, generated):
                answers = self._extract_answers(output)
                correct = sum(
                    1
                    for answer in answers
                    if answers_equivalent(answer, target_answer, self.grade_answer)
                )
                evidence = dict(results[index] or {})
                evidence.update(
                    {
                        "critic_correct_votes": correct,
                        "critic_valid_votes": len(answers),
                        "critic_total_votes": votes,
                        "answer_probability": correct / votes,
                    }
                )
                results[index] = evidence

        finalized: List[Dict[str, Any]] = []
        for index, item in enumerate(candidates):
            evidence = results[index]
            if evidence is None:
                evidence = _error_evidence(str(item.get("candidate_key", "")), "internal missing result")
            # Valid/semantic-invalid evidence is deterministic under the fixed
            # seed and can be reused.  Infrastructure errors are not cached so
            # an OOM or temporary missing file can be retried.
            if not evidence.get("error"):
                self._save_cache(evidence)
            finalized.append(evidence)
        return finalized


class RewardRequestHandler(BaseHTTPRequestHandler):
    evaluator: QuestionRewardEvaluator
    max_request_bytes: int

    def _write(self, status: int, value: Mapping[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            self._write(
                200,
                {
                    "ok": True,
                    "protocol_version": 1,
                    "answer_model": str(self.evaluator.args.answer_model),
                    "gpu_idle": not self.evaluator._answer_model_awake,
                },
            )
        else:
            self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/score":
            self._write(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_request_bytes:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if payload.get("protocol_version") != 1:
                raise ValueError("unsupported protocol_version")
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("candidates must be a list")
            results = self.evaluator.evaluate(candidates, int(payload.get("votes", 0)))
            self._write(200, {"protocol_version": 1, "results": results})
        except Exception as exc:
            LOGGER.exception("Reward request failed")
            self._write(400, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--processor-model", default=None)
    parser.add_argument("--image-model", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16"))
    # This service is scheduled in a separate phase from actor rollout/update;
    # the answer model sleeps and releases allocations before EasyQ1 resumes.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--votes", type=int, default=5)
    parser.add_argument("--min-agree-votes", type=int, default=2)
    parser.add_argument("--consensus-threshold", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1536 * 28 * 28)
    parser.add_argument("--image-device", default="cuda:0")
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--image-cpu-offload", action="store_true")
    parser.add_argument(
        "--unload-image-after-request",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete Qwen-Image-Edit after each reward batch to bound host RAM.",
    )
    parser.add_argument(
        "--sleep-answer-during-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Temporarily offload vLLM while Qwen-Image-Edit occupies the GPU.",
    )
    parser.add_argument("--image-num-inference-steps", type=int, default=20)
    parser.add_argument("--image-true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max-request-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.votes != 5:
        raise ValueError("This reward protocol fixes votes=5 so its probability floor is exactly 1/5.")
    if args.min_agree_votes < 2 or args.min_agree_votes > args.votes:
        raise ValueError("--min-agree-votes must be within [2, votes].")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    evaluator = QuestionRewardEvaluator(args)
    handler = RewardRequestHandler
    handler.evaluator = evaluator
    handler.max_request_bytes = args.max_request_bytes
    server = HTTPServer((args.host, args.port), handler)
    LOGGER.info("Question reward service listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping question reward service")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
