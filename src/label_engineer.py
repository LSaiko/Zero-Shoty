"""Validation and normalization helpers for zero-shot classifier label sets."""

import logging

logger = logging.getLogger(__name__)

MIN_LABELS = 2
MAX_LABELS = 20
MAX_LABEL_LENGTH = 60

# ponytail: known ceiling — only the most common antonym pairs are covered.
# Extend this dict if more pairs need warning coverage.
ANTONYM_PAIRS: dict[str, str] = {
    "true": "false",
    "false": "true",
    "yes": "no",
    "no": "yes",
    "good": "bad",
    "bad": "good",
    "positive": "negative",
    "negative": "positive",
}


class LabelEngineer:
    """Validates and normalizes candidate label sets for zero-shot classification."""

    def validate(self, labels: list[str]) -> tuple[bool, list[str]]:
        """Validate a candidate label set against structural and quality rules.

        Args:
            labels: Candidate labels to validate.

        Returns:
            A tuple of (is_valid, warnings) where is_valid is False on hard
            violations (label count, length, duplicates) and warnings is a
            list of non-fatal suggestions (single-word labels, missing
            neutral option for antonym pairs).
        """
        warnings: list[str] = []
        is_valid = True

        if len(labels) < MIN_LABELS:
            is_valid = False
            warnings.append(f"At least {MIN_LABELS} labels are required; got {len(labels)}")

        if len(labels) > MAX_LABELS:
            is_valid = False
            warnings.append(f"At most {MAX_LABELS} labels are allowed; got {len(labels)}")

        for label in labels:
            if len(label) > MAX_LABEL_LENGTH:
                is_valid = False
                warnings.append(f"Label '{label}' exceeds {MAX_LABEL_LENGTH} characters")

        seen: dict[str, str] = {}
        for label in labels:
            key = label.strip().lower()
            if key in seen:
                is_valid = False
                warnings.append(f"Duplicate label (case-insensitive): '{label}' and '{seen[key]}'")
            else:
                seen[key] = label

        for label in labels:
            stripped = label.strip()
            if stripped and " " not in stripped and len(stripped) <= MAX_LABEL_LENGTH:
                warnings.append(
                    f"Label '{label}' is a single word with no context — "
                    f"consider a contextual phrasing, e.g. 'positive sentiment'"
                )

        lower_labels = {label.strip().lower() for label in labels}
        for label_lower, antonym in ANTONYM_PAIRS.items():
            if label_lower in lower_labels and antonym in lower_labels:
                has_neutral = any(
                    keyword in lower_labels
                    for keyword in ("neutral", "uncertain", "unknown", "unclear")
                )
                if not has_neutral:
                    warnings.append(
                        f"Labels contain antonym pair '{label_lower}'/'{antonym}' "
                        f"without a neutral option — consider adding 'uncertain'"
                    )

        return is_valid, warnings

    def normalize(self, labels: list[str]) -> list[str]:
        """Normalize a label list by trimming whitespace and deduplicating.

        Strips surrounding whitespace, capitalizes only the first letter of
        each label (leaving the rest unchanged), and removes case-insensitive
        duplicates while preserving the original order.

        Args:
            labels: Raw candidate labels.

        Returns:
            Normalized, deduplicated list of labels.
        """
        normalized: list[str] = []
        seen: set[str] = set()

        for label in labels:
            stripped = label.strip()
            if not stripped:
                continue
            capitalized = stripped[0].upper() + stripped[1:] if stripped else stripped
            key = capitalized.lower()
            if key not in seen:
                seen.add(key)
                normalized.append(capitalized)

        return normalized

    def format_for_display(self, labels: list[str]) -> str:
        """Format a label list as a comma-separated display string.

        Args:
            labels: Labels to format.

        Returns:
            Comma-separated string of labels.
        """
        return ", ".join(labels)
