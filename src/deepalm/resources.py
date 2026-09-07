"""Cooperative resource guards for bounded Deep ALM workflow stages."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ResourceSnapshot:
    """Observed resource values at a cooperative stage boundary."""

    elapsed_seconds: float
    process_rss_bytes: int | None
    accelerator_allocated_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["process_rss_status"] = (
            "measured" if self.process_rss_bytes is not None else "unavailable"
        )
        result["accelerator_allocated_status"] = (
            "measured"
            if self.accelerator_allocated_bytes is not None
            else "unavailable"
        )
        return result


class BudgetExceeded(RuntimeError):
    """Raised when a configured cooperative resource guard is crossed."""

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class ResourceMonitor:
    """Measure and enforce independent wall-clock, RSS and accelerator limits."""

    def __init__(
        self,
        *,
        wall_clock_budget_seconds: float,
        process_rss_limit_bytes: int,
        accelerator_memory_limit_bytes: int,
        device: str = "cpu",
        clock: Callable[[], float] = time.monotonic,
        rss_reader: Callable[[], int | None] | None = None,
        accelerator_reader: Callable[[], int | None] | None = None,
    ) -> None:
        self._wall_clock_budget_seconds = wall_clock_budget_seconds
        self._process_rss_limit_bytes = process_rss_limit_bytes
        self._accelerator_memory_limit_bytes = accelerator_memory_limit_bytes
        self._clock = clock
        self._started_at = clock()
        self._rss_reader = rss_reader or _process_rss_bytes
        self._accelerator_reader = accelerator_reader or _accelerator_allocated_bytes(device)

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            elapsed_seconds=self._clock() - self._started_at,
            process_rss_bytes=self._rss_reader(),
            accelerator_allocated_bytes=self._accelerator_reader(),
        )

    def check(self, stage: str) -> ResourceSnapshot:
        """Raise a serializable failure at a safe boundary when a limit is exceeded."""

        snapshot = self.snapshot()
        observed = snapshot.to_dict()
        if snapshot.elapsed_seconds > self._wall_clock_budget_seconds:
            raise BudgetExceeded(
                f"wall-clock budget exceeded at {stage}",
                diagnostics={
                    "reason": "budget_exhausted",
                    "stage": stage,
                    "resource_snapshot": observed,
                    "limit_seconds": self._wall_clock_budget_seconds,
                },
            )
        if (
            snapshot.process_rss_bytes is not None
            and snapshot.process_rss_bytes > self._process_rss_limit_bytes
        ):
            raise BudgetExceeded(
                f"process RSS budget exceeded at {stage}",
                diagnostics={
                    "reason": "budget_exhausted",
                    "stage": stage,
                    "resource_snapshot": observed,
                    "limit_bytes": self._process_rss_limit_bytes,
                },
            )
        if (
            snapshot.accelerator_allocated_bytes is not None
            and snapshot.accelerator_allocated_bytes
            > self._accelerator_memory_limit_bytes
        ):
            raise BudgetExceeded(
                f"accelerator-memory budget exceeded at {stage}",
                diagnostics={
                    "reason": "budget_exhausted",
                    "stage": stage,
                    "resource_snapshot": observed,
                    "limit_bytes": self._accelerator_memory_limit_bytes,
                },
            )
        return snapshot


def _process_rss_bytes() -> int | None:
    """Return peak process RSS where the platform exposes an unambiguous unit."""

    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def _accelerator_allocated_bytes(device: str) -> Callable[[], int | None]:
    if device == "cuda":
        return lambda: int(torch.cuda.memory_allocated())
    return lambda: None
