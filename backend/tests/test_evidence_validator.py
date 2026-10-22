import pytest

from app.rag.evidence_validator import EvidenceValidator


def test_filters_irrelevant_chunks():
    validator = EvidenceValidator(max_distance=0.65)

    results = [
        {"text": "Relevant policy", "distance": 0.40},
        {"text": "Weak match", "distance": 0.70},
        {"text": "Unrelated text", "distance": 0.85},
    ]

    filtered = validator.filter_relevant_chunks(results)

    assert len(filtered) == 1
    assert filtered[0]["text"] == "Relevant policy"


def test_accepts_relevant_evidence():
    validator = EvidenceValidator(max_distance=0.65)

    results = [
        {"text": "Production access review", "distance": 0.45}
    ]

    assert validator.has_sufficient_evidence(results) is True


def test_rejects_irrelevant_evidence():
    validator = EvidenceValidator(max_distance=0.65)

    results = [
        {"text": "Unrelated policy", "distance": 0.75}
    ]

    assert validator.has_sufficient_evidence(results) is False


def test_empty_results():
    validator = EvidenceValidator(max_distance=0.65)

    assert validator.filter_relevant_chunks([]) == []
    assert validator.has_sufficient_evidence([]) is False


def test_invalid_threshold():
    with pytest.raises(ValueError):
        EvidenceValidator(max_distance=-0.1)
