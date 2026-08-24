"""Remove stale repository-owned Ray/vLLM children before an exclusive 8-GPU run.

This deliberately targets only Python processes from the repository-local
augmentation environment.  The working directory is used when available, but
Ray is allowed to change it to a session directory after startup.
It exists because vLLM V1 ``multiprocessing.spawn`` children can outlive a Ray
worker or reward HTTP parent after an interrupted run and retain tens of GB of
host RAM.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: str
    executable: Path
    cwd: Path
    environment: str = ""


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_process(pid: int) -> ProcessInfo | None:
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        # Fields after the final ')' start with state (field 3), then PPID.
        ppid = int(stat[stat.rfind(")") + 2 :].split()[1])
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        executable = (proc / "exe").resolve(strict=True)
        cwd = (proc / "cwd").resolve(strict=True)
        environment = (proc / "environ").read_bytes().replace(b"\0", b"\n").decode(
            "utf-8", errors="replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return None
    return ProcessInfo(pid, ppid, command, executable, cwd, environment)


def _snapshot() -> Dict[int, ProcessInfo]:
    processes: Dict[int, ProcessInfo] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = _read_process(int(entry.name))
        if info is not None:
            processes[info.pid] = info
    return processes


def _nvidia_compute_pids() -> Set[int]:
    """Return PIDs with an active NVIDIA compute context, if available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()
    pids: Set[int] = set()
    for value in result.stdout.splitlines():
        try:
            pids.add(int(value.strip()))
        except ValueError:
            continue
    return pids


def select_stale_processes(
    processes: Dict[int, ProcessInfo],
    repo_root: Path,
    environment_dir: Path,
    gpu_pids: Set[int] | None = None,
) -> Set[int]:
    gpu_pids = gpu_pids or set()
    protected: Set[int] = {os.getpid()}
    parent = os.getppid()
    while parent > 1 and parent not in protected:
        protected.add(parent)
        info = processes.get(parent)
        if info is None:
            break
        parent = info.ppid

    roots: Set[int] = set()
    for pid, info in processes.items():
        if pid in protected:
            continue
        # A venv's /proc/<pid>/exe commonly resolves to /usr/bin/python even
        # though argv[0] is the repository-local interpreter.
        from_local_python = (
            _inside(info.executable, environment_dir)
            or str(environment_dir) in info.command
            or f"VIRTUAL_ENV={environment_dir}" in info.environment
        )
        reward_server = "data_augmentation.question_reward_server" in info.command
        orphanable_vllm_child = "multiprocessing.spawn" in info.command
        repository_ray_worker = any(
            marker in info.command
            for marker in (
                "ray::WorkerDict",
                "ray::Runner",
                "ray/_private/workers/default_worker.py",
                "ray\\_private\\workers\\default_worker.py",
            )
        )
        marked_repository_runtime = from_local_python and (
            reward_server or orphanable_vllm_child or repository_ray_worker
        )
        # Ray can change a worker's cwd and process title after it starts.  The
        # inherited VIRTUAL_ENV still proves ownership by this repository, and
        # an active NVIDIA context proves that the process can retain VRAM.
        local_gpu_holder = from_local_python and pid in gpu_pids
        if marked_repository_runtime or local_gpu_holder:
            roots.add(pid)

    # Include descendants so a reward server and its engine workers disappear
    # together.  Iterate to a fixed point because process trees are shallow but
    # not guaranteed to have a fixed number of levels.
    selected = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, info in processes.items():
            if pid not in protected and pid not in selected and info.ppid in selected:
                selected.add(pid)
                changed = True
    return selected


def _signal_all(pids: Iterable[int], sig: signal.Signals) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            continue


def cleanup(repo_root: Path, grace_seconds: float) -> List[int]:
    repo_root = repo_root.resolve(strict=True)
    environment_dir = (repo_root / ".venv-augmentation").resolve(strict=True)
    processes = _snapshot()
    selected = select_stale_processes(
        processes, repo_root, environment_dir, _nvidia_compute_pids()
    )
    if not selected:
        return []

    # Children first gives Python parents a chance to reap their engines.
    ordered = sorted(selected, reverse=True)
    _signal_all(ordered, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    remaining = set(selected)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = {pid for pid in remaining if (Path("/proc") / str(pid)).exists()}
    _signal_all(sorted(remaining, reverse=True), signal.SIGKILL)
    # CUDA context teardown is asynchronous. Do not report cleanup success
    # until NVIDIA no longer attributes VRAM to one of the selected PIDs.
    gpu_deadline = time.monotonic() + grace_seconds
    gpu_remaining = _nvidia_compute_pids().intersection(selected)
    while gpu_remaining and time.monotonic() < gpu_deadline:
        time.sleep(0.1)
        gpu_remaining = _nvidia_compute_pids().intersection(selected)
    if gpu_remaining:
        raise RuntimeError(
            "repository GPU processes survived SIGKILL: "
            + ",".join(map(str, sorted(gpu_remaining)))
        )
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--grace-seconds", type=float, default=5.0)
    args = parser.parse_args()
    removed = cleanup(args.repo_root, args.grace_seconds)
    if removed:
        print("Stopped stale repository GPU/Ray/vLLM processes: " + ",".join(map(str, removed)))
    else:
        print("No stale repository GPU/Ray/vLLM processes found.")


if __name__ == "__main__":
    main()
