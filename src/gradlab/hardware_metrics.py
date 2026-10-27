"""Bounded host diagnostics, sampled by the supervisor into its durable journal."""

from __future__ import annotations

import math
import shutil
import subprocess

import psutil


class HardwareSampler:
    """Sample host CPU/RAM and visible NVIDIA devices without another service writer.

    CPU uses deltas between samples. Missing sensors are omitted, never reported
    as zero. The supervisor's existing 15-second health cadence bounds work.
    """

    def __init__(self) -> None:
        self._cpu_ready = False
        self._nvidia_smi = shutil.which("nvidia-smi")

    def sample(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            cpu = psutil.cpu_percent(interval=None)
            if self._cpu_ready:
                metrics["system/cpu_utilization_percentage"] = cpu
            self._cpu_ready = True
            memory = psutil.virtual_memory()
            metrics["system/system_memory_usage_megabytes"] = (memory.total - memory.available) / 1024**2
            metrics["system/system_memory_usage_percentage"] = 100 * (1 - memory.available / memory.total)
        except (OSError, ValueError, psutil.Error):
            pass
        if self._nvidia_smi is None:
            return metrics
        try:
            result = subprocess.run(
                [self._nvidia_smi,
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired):
            return metrics
        if result.returncode:
            return metrics
        for line in result.stdout.splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) != 6 or not fields[0].isdigit():
                continue
            names = ("utilization_percentage", "memory_usage_megabytes", "memory_total_megabytes",
                     "power_usage_watts", "temperature_celsius")
            for name, raw in zip(names, fields[1:], strict=True):
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value) and value >= 0:
                    metrics[f"system/gpu/{fields[0]}/{name}"] = value
        return metrics
