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

import logging
import os

import gradio as gr

from src.claude_advisor import ClaudeAdvisor
from src.classifier import ZeroShotClassifier
from src.label_engineer import LabelEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MIN_LABELS = 2

classifier = ZeroShotClassifier(model_id="facebook/bart-large-mnli")
classifier.warmup()
label_engineer = LabelEngineer()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
advisor = ClaudeAdvisor(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
if advisor is None:
    logger.info("ANTHROPIC_API_KEY not set — Claude Label Advisor is disabled.")

EXAMPLES = [
    [
        "Researchers unveiled a new chip architecture that promises twice the "
        "performance per watt of current data center GPUs.",
        "technology, sports, politics, entertainment",
        False,
    ],
    [
        "A new study finds that just 20 minutes of daily walking is linked to "
        "a significant drop in cardiovascular disease risk.",
        "health, sports, technology, politics",
        False,
    ],
    [
        "Lawmakers passed a sweeping budget bill after weeks of tense "
        "negotiations between both chambers of the legislature.",
        "politics, business, technology, health",
        False,
    ],
    [
        "The award-winning drama swept the ceremony, taking home trophies for "
        "best picture, director, and lead actress.",
        "entertainment, sports, politics, technology",
        False,
    ],
    [
        "The underdog team clinched the championship in overtime, capping off "
        "a stunning run through the playoffs.",
        "sports, entertainment, politics, health",
        False,
    ],
]


def classify(
    text: str, labels_str: str, multi_label: bool
) -> tuple[dict[str, float], dict, str, str, dict, str]:
    """Classify input text against user-supplied labels and advise if ambiguous.

    Args:
        text: Text to classify.
        labels_str: Comma-separated candidate labels.
        multi_label: Whether to allow multiple labels to apply independently.

    Returns:
        Tuple of (label_scores, full_result_json, warnings_text, interpretation,
        suggested_labels_json, reasoning) for the Gradio output components.
    """
    empty_advisor: tuple[dict, str] = ({}, "")

    if not text or not text.strip():
        return {}, {}, "Please enter some text to classify.", "", *empty_advisor

    labels = [label.strip() for label in labels_str.split(",") if label.strip()]
    if len(labels) < MIN_LABELS:
        return {}, {}, "Please provide at least two comma-separated labels.", "", *empty_advisor

    is_valid, warnings = label_engineer.validate(labels)
    warnings_text = "\n".join(warnings)
    if not is_valid:
        return {}, {}, warnings_text or "Invalid label set.", "", *empty_advisor

    normalized_labels = label_engineer.normalize(labels)
    result = classifier.classify(text, normalized_labels, multi_label=multi_label)

    label_scores = dict(zip(result["labels"], result["scores"]))

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

    return label_scores, result, warnings_text, interpretation, suggested_labels, reasoning


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="Zero-Shot Text Classifier") as demo:
    gr.Markdown("# 🎯 Zero-Shot Text Classifier")
    gr.Markdown(
        "Classify any text against labels you define on the fly, with an "
        "optional Claude-powered advisor for ambiguous results."
    )

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

    classify_btn.click(
        fn=classify,
        inputs=[input_text, labels_input, multi_label_checkbox],
        outputs=[
            top_prediction,
            full_scores,
            label_warnings,
            interpretation_box,
            suggested_labels_box,
            reasoning_box,
        ],
    )


if __name__ == "__main__":
    demo.launch()
