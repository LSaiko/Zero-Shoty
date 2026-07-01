# Architecture

## Component Flow

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

## Notes
- Claude is invoked **only** on the ambiguous branch — never in the hot path.
- The validator is the sole trust boundary for user input.
- BART-MNLI runs locally (or HF Inference); Claude is a separate, optional call.
