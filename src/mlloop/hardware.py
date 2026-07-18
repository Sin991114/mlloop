"""GPU detection for the goal-time hardware consent gate.

MLLoop never trains models, but the agent it harnesses does — and whether
training may occupy the user's GPU (cost, thermals, shared workstations) is
the user's call, not the agent's. Detection is framework-free via nvidia-smi.
"""

from __future__ import annotations

import subprocess


def detect_gpus(timeout: float = 5.0) -> list[dict]:
    """NVIDIA GPUs visible to nvidia-smi; empty list when none or undetectable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            gpus.append({"name": parts[0], "memory": parts[1] if len(parts) > 1 else None})
    return gpus
