"""Verified frozen references for benchmark policies consumed by MM."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deepalm.policies import BMDatePolicy

_REFERENCE_NAME = re.compile(r"\.baseline\.([0-9a-f]{64})\.json$")


class BaselineReferenceError(RuntimeError):
    """Raised when a frozen benchmark reference cannot prove its identity."""


@dataclass(frozen=True)
class FrozenDateBenchmarkReference:
    """Read-only identity record for a selected BM^D checkpoint.

    The reference intentionally records a content hash rather than relying on a
    mutable filename.  MM must resolve this reference before it can consume the
    benchmark policy.
    """

    reference_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    reference_identity: str
    horizon_years: int
    configuration_identity: str
    data_identities: tuple[tuple[str, str], ...]

    @classmethod
    def freeze(cls, checkpoint_path: Path) -> FrozenDateBenchmarkReference:
        """Create a portable, content-addressed reference beside a checkpoint."""

        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("policy") != "BM^D":
            raise BaselineReferenceError("Only a selected BM^D checkpoint can be frozen")
        payload = {
            "format_version": 1,
            "policy": "BM^D",
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "horizon_years": checkpoint["horizon_years"],
            "configuration_identity": checkpoint["configuration_identity"],
            "data_identities": checkpoint["data_identities"],
        }
        serialized = _serialize_json(payload)
        identity = _sha256_bytes(serialized)
        reference_path = checkpoint_path.with_suffix(f".baseline.{identity}.json")
        _save_text(reference_path, serialized)
        return cls.load(reference_path)

    @classmethod
    def load(
        cls,
        reference_path: Path | None,
        *,
        expected_reference_identity: str | None = None,
    ) -> FrozenDateBenchmarkReference:
        """Load only a reference whose content hash is pinned by its filename or caller."""

        if reference_path is None:
            raise BaselineReferenceError("A frozen BM^D baseline reference is required")
        filename_identity = _identity_from_filename(reference_path)
        if (
            expected_reference_identity is not None
            and expected_reference_identity != filename_identity
        ):
            raise BaselineReferenceError(
                "Frozen BM^D reference identity does not match its filename"
            )
        expected_identity = filename_identity
        if (
            not reference_path.is_file()
            or _sha256(reference_path) != expected_identity
        ):
            raise BaselineReferenceError("Frozen BM^D reference identity verification failed")
        try:
            payload = json.loads(reference_path.read_text(encoding="utf-8"))
            relative_checkpoint = payload["checkpoint_path"]
            checkpoint_relative_path = Path(relative_checkpoint)
            if checkpoint_relative_path.is_absolute() or ".." in checkpoint_relative_path.parts:
                raise ValueError("checkpoint_path must be relative to the reference")
            data_identities = payload["data_identities"]
            if payload["policy"] != "BM^D" or not isinstance(data_identities, dict):
                raise ValueError("unexpected baseline reference format")
            return cls(
                reference_path=reference_path,
                checkpoint_path=reference_path.parent / relative_checkpoint,
                checkpoint_sha256=str(payload["checkpoint_sha256"]),
                reference_identity=expected_identity,
                horizon_years=int(payload["horizon_years"]),
                configuration_identity=str(payload["configuration_identity"]),
                data_identities=tuple(
                    sorted((str(key), str(value)) for key, value in data_identities.items())
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise BaselineReferenceError("Invalid frozen BM^D baseline reference") from error

    def load_policy(
        self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64
    ) -> BMDatePolicy:
        """Verify the checkpoint identity and return a non-trainable BM^D policy."""

        if (
            not self.checkpoint_path.is_file()
            or _sha256(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise BaselineReferenceError("Frozen BM^D baseline identity verification failed")
        checkpoint = _load_checkpoint(self.checkpoint_path)
        if (
            checkpoint.get("policy") != "BM^D"
            or checkpoint.get("horizon_years") != self.horizon_years
            or checkpoint.get("configuration_identity") != self.configuration_identity
            or checkpoint.get("data_identities") != dict(self.data_identities)
        ):
            raise BaselineReferenceError("Frozen BM^D baseline checkpoint identity mismatch")
        transitions = 12 * self.horizon_years
        metadata = checkpoint.get("policy_metadata")
        if not isinstance(metadata, dict) or metadata.get("decision_dates") != transitions:
            raise BaselineReferenceError("Frozen BM^D baseline has incompatible decision dates")
        policy = BMDatePolicy(transitions=transitions, device=device, dtype=dtype)
        policy.load_state_dict(checkpoint["policy_state"])
        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        return policy


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise BaselineReferenceError("Frozen BM^D baseline checkpoint cannot be loaded") from error
    if not isinstance(loaded, dict):
        raise BaselineReferenceError("Frozen BM^D baseline checkpoint has invalid content")
    return loaded


def _serialize_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _save_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as temporary:
        temporary.write(contents)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(contents: str) -> str:
    return hashlib.sha256(contents.encode()).hexdigest()


def _identity_from_filename(path: Path) -> str:
    match = _REFERENCE_NAME.search(path.name)
    if match is None:
        raise BaselineReferenceError(
            "Frozen BM^D reference filename must contain its content identity"
        )
    return match.group(1)
