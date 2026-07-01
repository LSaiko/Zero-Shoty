"""Tests for src.label_engineer.LabelEngineer."""

from src.label_engineer import LabelEngineer


def test_validate_minimum_labels() -> None:
    """validate() should mark fewer than 2 labels as invalid."""
    engineer = LabelEngineer()
    is_valid, warnings = engineer.validate(["only-one"])
    assert is_valid is False
    assert any(warnings)


def test_validate_maximum_labels() -> None:
    """validate() should mark more than 20 labels as invalid."""
    engineer = LabelEngineer()
    labels = [f"label {i}" for i in range(21)]
    is_valid, warnings = engineer.validate(labels)
    assert is_valid is False
    assert any(warnings)


def test_validate_duplicate_warning() -> None:
    """validate() should catch case-insensitive duplicate labels."""
    engineer = LabelEngineer()
    is_valid, warnings = engineer.validate(["positive sentiment", "Positive Sentiment"])
    assert is_valid is False
    assert any("duplicate" in w.lower() for w in warnings)


def test_validate_single_word_warning() -> None:
    """validate() should warn when labels are single words with no context."""
    engineer = LabelEngineer()
    is_valid, warnings = engineer.validate(["good", "bad"])
    assert any("single word" in w.lower() for w in warnings)


def test_normalize_strips_whitespace() -> None:
    """normalize() should strip surrounding whitespace from labels."""
    engineer = LabelEngineer()
    result = engineer.normalize(["  good  ", " bad"])
    assert result == ["Good", "Bad"]


def test_normalize_deduplicates() -> None:
    """normalize() should remove case-insensitive duplicates, preserving order."""
    engineer = LabelEngineer()
    result = engineer.normalize(["good", "Good", "GOOD", "bad"])
    assert result == ["Good", "Bad"]
