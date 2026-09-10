"""Shared invariants for the compact local-validation workflows."""

from __future__ import annotations


def is_valid_local_validation_selection_epoch(value: object) -> bool:
    """Return whether a manifest epoch is one of the two local training epochs.

    ``bool`` is deliberately excluded even though it is an ``int`` subclass: the
    value is an artifact-schema integer, not a truth value that happens to have
    an integer representation.
    """

    return type(value) is int and value in {1, 2}
