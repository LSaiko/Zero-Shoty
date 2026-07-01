"""Claude-based advisor that critiques ambiguous zero-shot predictions.

Claude's role here is strictly SYNTHESIZER / ADVISOR: it never classifies or
retrieves anything. It receives an ambiguous prediction and the label set,
and returns label critique + suggested rephrasing for a human (or the
caller) to consider.
"""

import json
import logging
import os
import time

import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a label engineering assistant for zero-shot NLP classifiers.
You receive: (1) a text sample, (2) a user-defined label set,
(3) a classification result where the model was uncertain.
Your job is to interpret the ambiguity and suggest 2-4 improved label phrasings
that would give the model a cleaner signal.
Respond ONLY in valid JSON matching this schema:
{interpretation, suggested_labels, reasoning, confidence}
Do not include markdown fences or preamble."""


class ClaudeAdvisor:
    """Advisor role wrapping the Claude API for label critique only."""

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Claude advisor client.

        Args:
            api_key: Anthropic API key. Falls back to the ANTHROPIC_API_KEY
                environment variable if not provided.

        Returns:
            None.
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_key)

    def advise(
        self, text: str, labels: list[str], prediction: dict, ambiguity_reason: str
    ) -> dict:
        """Ask Claude to interpret an ambiguous prediction and suggest better labels.

        Args:
            text: Original input text that was classified.
            labels: Candidate labels used for classification.
            prediction: Classification result dict from ZeroShotClassifier.classify.
            ambiguity_reason: Human-readable reason the prediction was flagged ambiguous.

        Returns:
            Dict with keys "interpretation", "suggested_labels", "reasoning",
            and "confidence". On any failure, returns a graceful fallback
            dict with the same keys.
        """
        try:
            user_content = (
                f"Text: {text}\n"
                f"Labels: {labels}\n"
                f"Prediction: {prediction}\n"
                f"Ambiguity reason: {ambiguity_reason}"
            )

            start = time.perf_counter()
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            elapsed = time.perf_counter() - start
            logger.debug("Claude advisor call took %.3fs", elapsed)

            parsed = json.loads(response.content[0].text)
            return {
                "interpretation": parsed.get("interpretation"),
                "suggested_labels": parsed.get("suggested_labels", []),
                "reasoning": parsed.get("reasoning"),
                "confidence": parsed.get("confidence"),
            }
        except Exception as exc:
            logger.error("Claude advisor call failed: %s", exc)
            return {
                "interpretation": "Claude API unavailable",
                "suggested_labels": [],
                "reasoning": str(exc),
                "confidence": "low",
            }
