"""Tests for src.classifier.ZeroShotClassifier.

All transformers pipeline calls are mocked; no model download or real
inference occurs.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.classifier import ZeroShotClassifier


@pytest.fixture
def mock_classifier():
    """Build a ZeroShotClassifier with the transformers pipeline mocked out."""
    with patch("src.classifier.pipeline") as mock_pipeline_factory:
        mock_pipeline_instance = MagicMock()
        mock_pipeline_factory.return_value = mock_pipeline_instance
        classifier = ZeroShotClassifier()
        yield classifier, mock_pipeline_instance


def test_classify_returns_correct_keys(mock_classifier) -> None:
    """classify() should return a dict with all expected keys."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.9, 0.1],
    }

    result = classifier.classify("some text", ["a", "b"])

    assert set(result.keys()) == {
        "labels",
        "scores",
        "top_label",
        "top_score",
        "is_ambiguous",
        "ambiguity_reason",
    }


def test_ambiguity_low_score(mock_classifier) -> None:
    """A top score of 0.50 should mark the result as ambiguous."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.50, 0.30],
    }

    result = classifier.classify("some text", ["a", "b"])

    assert result["is_ambiguous"] is True


def test_ambiguity_close_scores(mock_classifier) -> None:
    """Top two scores of 0.52 and 0.48 should mark the result as ambiguous."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.52, 0.48],
    }

    result = classifier.classify("some text", ["a", "b"])

    assert result["is_ambiguous"] is True


def test_not_ambiguous(mock_classifier) -> None:
    """A high top score with a clear gap should not be ambiguous."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.91, 0.05],
    }

    result = classifier.classify("some text", ["a", "b"])

    assert result["is_ambiguous"] is False


def test_single_label_has_no_active_labels(mock_classifier) -> None:
    """Single-label mode should not include an 'active_labels' key."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.9, 0.1],
    }

    result = classifier.classify("some text", ["a", "b"])

    assert "active_labels" not in result


def test_multi_label_active_labels(mock_classifier) -> None:
    """Multi-label mode returns every label scoring at or above 0.40."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b", "c", "d"],
        "scores": [0.95, 0.62, 0.40, 0.12],
    }

    result = classifier.classify("some text", ["a", "b", "c", "d"], multi_label=True)

    assert result["active_labels"] == ["a", "b", "c"]
    mock_pipeline_instance.assert_called_once_with(
        "some text", ["a", "b", "c", "d"], multi_label=True
    )


def test_multi_label_no_active_labels(mock_classifier) -> None:
    """Multi-label mode yields an empty list when nothing clears the threshold."""
    classifier, mock_pipeline_instance = mock_classifier
    mock_pipeline_instance.return_value = {
        "labels": ["a", "b"],
        "scores": [0.30, 0.10],
    }

    result = classifier.classify("some text", ["a", "b"], multi_label=True)

    assert result["active_labels"] == []
