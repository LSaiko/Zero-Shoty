"""Zero-shot text classifier with a Claude-powered label advisor.

---
title: Zero-Shot Text Classifier
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.x
app_file: app.py
pinned: false
license: mit
---
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

import gradio as gr

from src.claude_advisor import ClaudeAdvisor
from src.classifier import ZeroShotClassifier
from src.label_engineer import LabelEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_LABELS = 2

NER_API_URL = os.environ.get("NER_API_URL")
# Entity type -> auto-generated classification label.
ENTITY_LABEL_MAP = {
    "PER": "person-focused",
    "ORG": "organization news",
    "LOC": "location-based",
    "DATE": "time-sensitive event",
}

classifier = ZeroShotClassifier(model_id="facebook/bart-large-mnli")
classifier.warmup()
label_engineer = LabelEngineer()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
advisor = ClaudeAdvisor(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
if advisor is None:
    logger.info("ANTHROPIC_API_KEY not set — Claude Label Advisor is disabled.")

# Real snippets from the Kaggle News Category Dataset (heegyu/news-category-dataset).
# Each demonstrates a distinct ambiguity scenario for the classifier + advisor.
EXAMPLES = [
    # Overlapping topics: a tech-platform story that is equally about crime/business.
    [
        "Twitch Bans Gambling Sites After Streamer Scams Folks Out Of $200,000. "
        "One man's claims that he scammed people on the platform caused several "
        "popular streamers to consider a Twitch boycott.",
        "technology, business, crime, entertainment",
        False,
    ],
    # Multi-label case: a story that genuinely belongs to two categories at once.
    [
        "Maury Wills, Base-Stealing Shortstop For Dodgers, Dies At 89. He helped "
        "the Los Angeles Dodgers win three World Series titles with his "
        "base-stealing prowess.",
        "sports, obituary, entertainment, politics",
        True,
    ],
    # Politics vs. world news: the same event fits both framings.
    [
        "Biden Says U.S. Forces Would Defend Taiwan If China Invaded. President "
        "issues vow as tensions with China rise.",
        "politics, world news, business, technology",
        False,
    ],
    # Poor label set: none of the candidates cleanly fit (should trigger advice).
    [
        "How A New Documentary Captures The Complexity Of Being A Child Of "
        "Immigrants. Director Isabel Castro blends music documentary with the "
        "style of 'Euphoria' to tell a personal story.",
        "sports, finance, weather, technology",
        False,
    ],
    # Near-synonym labels: health vs. science vs. general news blur together.
    [
        "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID "
        "Boosters. Experts said it is too early to predict whether demand would "
        "match the 171 million doses of the new booster.",
        "health, science, u.s. news, politics",
        False,
    ],
]


def classify(
    text: str, labels_str: str, multi_label: bool
) -> tuple[dict[str, float], gr.CheckboxGroup, dict, str, str, dict, str]:
    """Classify input text against user-supplied labels and advise if ambiguous.

    Args:
        text: Text to classify.
        labels_str: Comma-separated candidate labels.
        multi_label: Whether to allow multiple labels to apply independently.

    Returns:
        Tuple of (label_scores, active_labels_checklist, full_result_json,
        warnings_text, interpretation, suggested_labels_json, reasoning) for the
        Gradio output components.
    """
    empty_checklist = gr.CheckboxGroup(choices=[], value=[], visible=False)
    empty_advisor: tuple[dict, str] = ({}, "")

    if not text or not text.strip():
        return {}, empty_checklist, {}, "Please enter some text to classify.", "", *empty_advisor

    labels = [label.strip() for label in labels_str.split(",") if label.strip()]
    if len(labels) < MIN_LABELS:
        return (
            {},
            empty_checklist,
            {},
            "Please provide at least two comma-separated labels.",
            "",
            *empty_advisor,
        )

    is_valid, warnings = label_engineer.validate(labels)
    warnings_text = "\n".join(warnings)
    if not is_valid:
        return {}, empty_checklist, {}, warnings_text or "Invalid label set.", "", *empty_advisor

    normalized_labels = label_engineer.normalize(labels)
    result = classifier.classify(text, normalized_labels, multi_label=multi_label)

    label_scores = dict(zip(result["labels"], result["scores"]))

    if multi_label:
        active = result.get("active_labels", [])
        active_checklist = gr.CheckboxGroup(
            choices=active, value=active, visible=True, label="Active Labels (score ≥ 0.40)"
        )
    else:
        active_checklist = empty_checklist

    interpretation, suggested_labels, reasoning = "", {}, ""
    if result["is_ambiguous"]:
        if advisor is not None:
            advice = advisor.advise(
                text, normalized_labels, result, result["ambiguity_reason"]
            )
            interpretation = advice["interpretation"]
            suggested_labels = advice["suggested_labels"]
            reasoning = advice["reasoning"]
        else:
            gr.Info("Set ANTHROPIC_API_KEY to enable Claude Label Advisor")
            interpretation = "Claude Label Advisor is disabled (no API key set)."

    return (
        label_scores,
        active_checklist,
        result,
        warnings_text,
        interpretation,
        suggested_labels,
        reasoning,
    )


def _entity_type(entity: dict) -> str | None:
    """Extract a normalized entity type (PER/ORG/LOC/DATE) from an NER result item."""
    raw = entity.get("entity_group") or entity.get("entity") or entity.get("type") or ""
    raw = raw.split("-")[-1].upper()  # strip B-/I- prefixes
    return raw if raw in ENTITY_LABEL_MAP else None


def classify_with_ner(text: str, multi_label: bool) -> tuple[list, dict, str]:
    """Extract entities via the NER service, build a label set, and classify.

    Args:
        text: Text to classify.
        multi_label: Whether labels are independent rather than mutually exclusive.

    Returns:
        Tuple of (ner_entities, label_scores, status_message).
    """
    if not NER_API_URL:
        return [], {}, "NER_API_URL is not set — this tab is disabled."
    if not text or not text.strip():
        return [], {}, "Please enter some text to classify."

    try:
        req = urllib.request.Request(
            NER_API_URL,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("NER request failed: %s", exc)
        return [], {}, f"NER service call failed: {exc}"

    entities = payload.get("entities", payload) if isinstance(payload, dict) else payload
    types = {t for e in entities if (t := _entity_type(e))}
    labels = [ENTITY_LABEL_MAP[t] for t in ENTITY_LABEL_MAP if t in types]

    if len(labels) < MIN_LABELS:
        return entities, {}, (
            f"Found {len(labels)} usable entity type(s); need at least {MIN_LABELS} to classify."
        )

    result = classifier.classify(text, labels, multi_label=multi_label)
    label_scores = dict(zip(result["labels"], result["scores"]))
    return entities, label_scores, f"Auto-generated labels: {', '.join(labels)}"


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Zero-Shot Text Classifier") as demo:
    gr.Markdown("# 🎯 Zero-Shot Text Classifier")
    gr.Markdown(
        "Classify any text against labels you define on the fly, with an "
        "optional Claude-powered advisor for ambiguous results."
    )

    with gr.Tab("Classification"):
      with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text", lines=4, placeholder="Paste any text to classify..."
            )
            labels_input = gr.Textbox(
                label="Labels (comma-separated)",
                placeholder="e.g. technology, sports, politics, entertainment",
            )
            multi_label_checkbox = gr.Checkbox(label="Multi-label mode", value=False)
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column():
            top_prediction = gr.Label(label="Top Prediction", num_top_classes=5)
            active_labels_group = gr.CheckboxGroup(
                label="Active Labels (score ≥ 0.40)", choices=[], visible=False
            )
            full_scores = gr.JSON(label="Full Scores")
            label_warnings = gr.Textbox(label="Label Warnings", lines=2, interactive=False)

            with gr.Accordion("Claude Label Advisor", open=False):
                interpretation_box = gr.Textbox(
                    label="Interpretation", lines=2, interactive=False
                )
                suggested_labels_box = gr.JSON(label="Suggested Labels")
                reasoning_box = gr.Textbox(label="Reasoning", lines=2, interactive=False)

      gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, labels_input, multi_label_checkbox],
      )

    with gr.Tab("Entity-Aware Classification"):
        if not NER_API_URL:
            gr.Markdown(
                "⚠️ **Disabled.** Set the `NER_API_URL` environment variable to the "
                "ner-extraction-tool FastAPI endpoint to enable entity-aware classification."
            )
        gr.Markdown(
            "Extracts named entities via the ner-extraction-tool, auto-generates a "
            "label set from the entity types (PER/ORG/LOC/DATE), then classifies."
        )
        with gr.Row():
            with gr.Column():
                ner_input_text = gr.Textbox(
                    label="Input Text", lines=4, placeholder="Paste any text to classify..."
                )
                ner_multi_label = gr.Checkbox(label="Multi-label mode", value=False)
                ner_btn = gr.Button(
                    "Extract Entities & Classify",
                    variant="primary",
                    interactive=bool(NER_API_URL),
                )
            with gr.Column():
                ner_status = gr.Textbox(label="Status", lines=2, interactive=False)
                with gr.Row():
                    ner_entities = gr.JSON(label="NER Entities")
                    ner_scores = gr.Label(label="Classification", num_top_classes=5)

        ner_btn.click(
            fn=classify_with_ner,
            inputs=[ner_input_text, ner_multi_label],
            outputs=[ner_entities, ner_scores, ner_status],
        )

    classify_btn.click(
        fn=classify,
        inputs=[input_text, labels_input, multi_label_checkbox],
        outputs=[
            top_prediction,
            active_labels_group,
            full_scores,
            label_warnings,
            interpretation_box,
            suggested_labels_box,
            reasoning_box,
        ],
    )


if __name__ == "__main__":
    demo.launch()
