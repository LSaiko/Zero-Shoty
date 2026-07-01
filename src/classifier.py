"""Zero-shot text classification wrapping facebook/bart-large-mnli."""

import logging
import time

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

LOW_SCORE_THRESHOLD = 0.55
SCORE_GAP_THRESHOLD = 0.12
ACTIVE_LABEL_THRESHOLD = 0.40


class ZeroShotClassifier:
    """Zero-shot classifier built on a Hugging Face `zero-shot-classification` pipeline."""

    def __init__(self, model_id: str = "facebook/bart-large-mnli") -> None:
        """Load the zero-shot-classification pipeline for the given model.

        Args:
            model_id: Hugging Face model identifier to load.

        Returns:
            None.
        """
        device = 0 if torch.cuda.is_available() else -1
        start = time.perf_counter()
        self.pipeline = pipeline("zero-shot-classification", model=model_id, device=device)
        elapsed = time.perf_counter() - start
        logger.info("Loaded model '%s' on device %s in %.2fs", model_id, device, elapsed)

    def classify(self, text: str, labels: list[str], multi_label: bool = False) -> dict:
        """Classify text against a set of candidate labels.

        Args:
            text: The input text to classify.
            labels: Candidate labels to score the text against.
            multi_label: Whether labels are independent (multi-label) rather than mutually exclusive.

        Returns:
            A dict with keys "labels", "scores", "top_label", "top_score",
            "is_ambiguous", "ambiguity_reason", and (multi-label only)
            "active_labels".
        """
        result = self.pipeline(text, labels, multi_label=multi_label)
        scores = result["scores"]
        result_labels = result["labels"]

        top_label = result_labels[0]
        top_score = scores[0]

        is_ambiguous = False
        ambiguity_reason = None

        if top_score < LOW_SCORE_THRESHOLD:
            is_ambiguous = True
            ambiguity_reason = f"Top score {top_score:.2f} below {LOW_SCORE_THRESHOLD} threshold"
        elif len(scores) >= 2 and (scores[0] - scores[1]) < SCORE_GAP_THRESHOLD:
            is_ambiguous = True
            ambiguity_reason = (
                f"Top two scores within {SCORE_GAP_THRESHOLD} "
                f"({scores[0]:.2f} vs {scores[1]:.2f})"
            )

        output = {
            "labels": result_labels,
            "scores": scores,
            "top_label": top_label,
            "top_score": top_score,
            "is_ambiguous": is_ambiguous,
            "ambiguity_reason": ambiguity_reason,
        }

        if multi_label:
            output["active_labels"] = [
                label
                for label, score in zip(result_labels, scores)
                if score >= ACTIVE_LABEL_THRESHOLD
            ]

        return output

    def warmup(self) -> None:
        """Run a dummy classification to force model weights to load.

        Args:
            None.

        Returns:
            None.
        """
        self.classify("warmup", ["a", "b"])
        logger.info("Warmup classification complete")
