#!/usr/bin/env python3
"""Validate the standalone Qwen augmentation and GRPO compatibility stack."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import subprocess
import sys
from typing import Dict, List

from packaging.version import Version


EXPECTED: Dict[str, str] = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "torchaudio": "2.6.0",
    "vllm": "0.8.3",
    "transformers": "4.51.3",
    "diffusers": "0.37.1",
    "qwen-vl-utils": "0.0.10",
    "numpy": "1.26.4",
    "ray": "2.43.0",
    "datasets": "3.5.0",
    "pyarrow": "19.0.1",
    "pandas": "2.2.3",
    "omegaconf": "2.3.0",
    "torchdata": "0.11.0",
    "tensordict": "0.7.2",
    "codetiming": "1.4.0",
    "jinja2": "3.1.6",
    "tensorboard": "2.19.0",
    "wandb": "0.19.11",
}

KERNEL_EXPECTED: Dict[str, str] = {
    "flash-attn": "2.7.4.post1",
    "flashinfer-python": "0.2.2.post1",
}


def check_versions(expected: Dict[str, str], errors: List[str]) -> None:
    for distribution, required in expected.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            errors.append(f"missing distribution: {distribution}=={required}")
            continue
        # Ignore only a wheel's local build suffix (for example +cu124), while
        # retaining post releases such as 2.7.4.post1 in the comparison.
        if Version(installed).public != Version(required).public:
            errors.append(f"{distribution}: installed {installed}, required {required}")


def import_runtime(errors: List[str], check_kernels: bool) -> None:
    try:
        import numpy  # noqa: F401
        import torch
        import torchaudio  # noqa: F401
        import torchvision  # noqa: F401
        import datasets  # noqa: F401
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
        import tensordict  # noqa: F401
        from codetiming import Timer  # noqa: F401
        from diffusers import QwenImageEditPlusPipeline  # noqa: F401
        from mathruler.grader import extract_boxed_content, grade_answer  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
        from qwen_vl_utils import process_vision_info  # noqa: F401
        from torchdata.stateful_dataloader import StatefulDataLoader  # noqa: F401
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        import wandb  # noqa: F401
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F401
        from vllm import LLM, SamplingParams  # noqa: F401
    except Exception as exc:  # binary/ABI failures are not always ImportError
        errors.append(f"runtime import failed: {exc!r}")
        return

    if torch.version.cuda != "12.4":
        errors.append(
            f"torch CUDA runtime is {torch.version.cuda!r}; required '12.4' "
            "(install the official cu124 wheel, not a CPU or different-CUDA wheel)"
        )

    if check_kernels:
        try:
            import flash_attn  # noqa: F401
            import flashinfer  # noqa: F401
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss  # noqa: F401
        except Exception as exc:
            errors.append(f"training kernel import/ABI check failed: {exc!r}")


def check_gpus(
    required_gpus: int,
    errors: List[str],
    details: List[str],
    gpu_profile: str,
) -> None:
    if required_gpus <= 0:
        return
    try:
        import torch

        try:
            driver_output = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            driver_versions = sorted({line.strip() for line in driver_output.splitlines() if line.strip()})
            if not driver_versions:
                errors.append("nvidia-smi returned no driver version")
            for driver_version in driver_versions:
                if Version(driver_version) < Version("550.54.14"):
                    errors.append(
                        f"NVIDIA driver {driver_version} is older than 550.54.14, "
                        "the CUDA 12.4 GA Linux baseline"
                    )
            if driver_versions:
                details.append("NVIDIA driver=" + ",".join(driver_versions))
        except Exception as exc:
            errors.append(f"unable to validate NVIDIA driver with nvidia-smi: {exc!r}")

        if not torch.cuda.is_available():
            errors.append("CUDA is not available to PyTorch")
            return
        count = torch.cuda.device_count()
        if count < required_gpus:
            errors.append(f"visible CUDA devices: {count}, required: {required_gpus}")
            return
        for index in range(required_gpus):
            properties = torch.cuda.get_device_properties(index)
            if gpu_profile == "a800":
                if "A800" not in properties.name.upper():
                    errors.append(
                        f"cuda:{index} is {properties.name!r}; the a800 profile requires NVIDIA A800"
                    )
                if properties.major < 8:
                    errors.append(
                        f"cuda:{index} ({properties.name}) has compute capability "
                        f"{properties.major}.{properties.minor}; Ampere (8.0+) is required"
                    )
                if properties.total_memory < 75 * 1024**3:
                    errors.append(
                        f"cuda:{index} ({properties.name}) exposes only "
                        f"{properties.total_memory / 1024**3:.1f}GiB; "
                        "the A800 profile requires the 80GB class"
                    )
            else:
                if properties.major < 8:
                    errors.append(
                        f"cuda:{index} ({properties.name}) has compute capability "
                        f"{properties.major}.{properties.minor}; Ampere (8.0+) is required"
                    )
                if properties.total_memory < 75 * 1024**3:
                    errors.append(
                        f"cuda:{index} ({properties.name}) exposes only "
                        f"{properties.total_memory / 1024**3:.1f}GiB; "
                        "the generic profile requires at least 75GiB"
                    )
            # Allocating and synchronizing catches driver/runtime incompatibility
            # before the eight expensive vLLM processes are launched.
            torch.empty(1, device=f"cuda:{index}")
            torch.cuda.synchronize(index)
            details.append(
                f"cuda:{index}={properties.name} "
                f"cc={properties.major}.{properties.minor} "
                f"memory={properties.total_memory / 1024**3:.1f}GiB"
            )
        if not torch.cuda.is_bf16_supported():
            errors.append("the visible CUDA stack does not report bfloat16 support")
    except Exception as exc:
        errors.append(f"CUDA initialization failed: {exc!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--gpu-profile", choices=("generic", "a800"), default="generic")
    parser.add_argument("--skip-training-kernels", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors: List[str] = []
    details: List[str] = []

    if sys.version_info[:2] != (3, 10):
        errors.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is active; "
            "the reproducible augmentation binary stack requires Python 3.10"
        )

    check_versions(EXPECTED, errors)
    if not args.skip_training_kernels:
        check_versions(KERNEL_EXPECTED, errors)
    import_runtime(errors, check_kernels=not args.skip_training_kernels)
    check_gpus(args.require_gpus, errors, details, args.gpu_profile)

    if errors:
        print("Standalone Qwen augmentation/GRPO runtime check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if not args.quiet:
        versions = {name: metadata.version(name) for name in EXPECTED}
        if not args.skip_training_kernels:
            versions.update({name: metadata.version(name) for name in KERNEL_EXPECTED})
        print("Standalone Qwen augmentation/GRPO runtime check OK")
        print("Python: " + sys.executable)
        print("Versions: " + ", ".join(f"{name}={version}" for name, version in versions.items()))
        for detail in details:
            print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
