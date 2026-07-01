---
title: Zero-Shot Text Classifier
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Zero-Shot Text Classifier

This Space classifies any text against label sets you define on the fly, using
`facebook/bart-large-mnli` for zero-shot natural language inference — no
training data or fine-tuning required. Enter text, list your candidate
labels, and get a ranked confidence score for each one, with optional
multi-label scoring for texts that fit more than one category.

When a prediction comes back ambiguous (low confidence or a narrow margin
between the top labels), the app can consult Claude as a label-engineering
advisor: Claude never classifies the text itself, but reviews the ambiguous
result and the label set to suggest clearer labels or explain why the text
is a hard case. Set `ANTHROPIC_API_KEY` as a Space Secret to enable Claude
Label Advisor. Source and documentation: https://github.com/LSaiko/Zero-shotty
