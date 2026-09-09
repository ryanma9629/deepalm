"""Developer-facing documentation contracts for corrected financial semantics."""

from __future__ import annotations

from pathlib import Path


def test_corrected_semantics_docs_expose_one_current_execution_contract() -> None:
    repository = Path(__file__).resolve().parents[1]
    readme = (repository / "README.md").read_text(encoding="utf-8")
    errata_path = repository / "docs/implementation-errata.md"

    assert "corrected-pilot" in readme
    assert "corrected-evaluate" in readme
    assert "corrected-report" in readme
    assert "paired-pilot" not in readme
    assert "paired-evaluate" not in readme
    assert "paired-report" not in readme
    assert not (repository / "docs/financial-corrections.md").exists()

    errata = errata_path.read_text(encoding="utf-8")
    for required_section in ("已采纳修正", "明确简化", "能力缺口"):
        assert required_section in errata
    assert "docs/errata.pdf" in errata
    assert "影响边界" in errata
    assert "公开验证" in errata
    assert "Corrected Financial Semantics" in errata
