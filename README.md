# Zero-Shoty

Zero-shot text classification with a Claude-powered label advisor for ambiguous predictions.

![CI](https://github.com/LSaiko/Zero-Shoty/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
[![🤗 HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HF%20Spaces-yellow)](#)

## About
Zero-shot classification lets a model assign arbitrary, user-defined labels to text at
inference time by reframing classification as Natural Language Inference (NLI): the input
text is the premise, each candidate label becomes a hypothesis, and `facebook/bart-large-mnli`
scores how well each hypothesis is entailed by the premise. This means new label sets work
immediately, with no training data or fine-tuning step. Claude is layered on top as a
label-engineering advisor — it never classifies text itself, but when a prediction comes back
ambiguous, it reviews the label set and the classifier's scores to suggest clearer labels or
explain why the case is genuinely hard, keeping the expensive LLM call out of the classification
hot path.

## Live Demo
[🤗 Try on HuggingFace Spaces](#)

## Skills Demonstrated
| Skill | Implementation |
|-------|-----------------|
| Zero-shot NLI classification | `src/classifier.py` |
| NLI mechanics (entailment scoring) | `src/classifier.py`, `docs/interview_qa.md` |
| Gradio UI | `app.py` |
| Prompt engineering | `src/claude_advisor.py` |
| Label engineering | `src/label_engineer.py` |
| Claude API integration | `src/claude_advisor.py` |
| HF Spaces deploy | `spaces/README.md` |
| CI/CD | `.github/workflows/ci.yml` |

## Architecture

```
┌─────────────┐
│ User Input  │  text + candidate labels (via Gradio UI)
└──────┬──────┘
       ▼
┌──────────────────────┐
│ Label Set Validator  │  dedupe, strip, reject empty / >N labels
│ (label_engineer.py)  │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ BART-MNLI Pipeline   │  facebook/bart-large-mnli
│ (classifier.py)      │  entailment scoring per label
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│ Score Threshold      │  top score < threshold OR
│ Check                │  top-2 gap < margin ?
└──────┬───────────────┘
       │
   ┌───┴───────────────┐
   │                   │
 confident          ambiguous
   │                   ▼
   │         ┌──────────────────────┐
   │         │ Claude Advisor       │  ambiguous output + label set
   │         │ (claude_advisor.py)  │  → critique + suggestions
   │         └──────┬───────────────┘
   │                ▼
   │         ┌──────────────────────┐
   │         │ Label Suggestions    │
   │         └──────┬───────────────┘
   ▼                ▼
┌──────────────────────┐
│ Gradio UI Output     │  scores table + (optional) advisor notes
│ (app.py)             │
└──────────────────────┘
```

Notes:
- Claude is invoked **only** on the ambiguous branch — never in the hot path.
- The validator is the sole trust boundary for user input.
- BART-MNLI runs locally (or HF Inference); Claude is a separate, optional call.

See [docs/architecture.md](docs/architecture.md) for the source of this diagram.

## Installation
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

## Usage
```powershell
python app.py
```
Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

Examples sourced from Kaggle News Category Dataset.

## Project Structure
```
Zero-Shoty/
├── CLAUDE.md
├── README.md
├── app.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── label_engineer.py
│   └── claude_advisor.py
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py
│   └── test_label_engineer.py
├── docs/
│   ├── architecture.md
│   └── interview_qa.md
└── spaces/
    └── README.md
```

## How It Works
1. **Classify** — input text and comma-separated labels are validated, normalized, and
   scored by `ZeroShotClassifier` (BART-MNLI) via entailment.
2. **Ambiguity check** — the classifier flags results with a low top score or a narrow
   margin between the top two labels as ambiguous.
3. **Claude advisor** — if ambiguous and an API key is configured, `ClaudeAdvisor` reviews
   the text, labels, and scores to suggest better labels and explain the ambiguity.
4. **Output** — the Gradio UI renders the top prediction, full score breakdown, label
   warnings, and (when triggered) Claude's interpretation, suggested labels, and reasoning.

## Interview Q&A
See [docs/interview_qa.md](docs/interview_qa.md) for detailed NLI theory, label
engineering, and productionization discussion.

## Contributing
1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
4. Open a pull request against `main`.

## License
MIT — © LSaiko. See [LICENSE](LICENSE).
