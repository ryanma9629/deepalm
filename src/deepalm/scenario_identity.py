"""Stable market-path identities independent of batch partitioning."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioIdentity:
    """Name one scenario without relying on worker order or device count."""

    split: str
    epoch: int
    seed: int
    global_path_index: int

    def derived_seed(self) -> int:
        """Return a reproducible RNG seed for this exact path identity."""

        encoded = f"{self.split}|{self.epoch}|{self.seed}|{self.global_path_index}".encode()
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
