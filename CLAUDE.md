# CLAUDE.md — Zero-Shoty

## 1. Project Identity
- **Repo:** LSaiko/Zero-Shoty
- **Stack:** `facebook/bart-large-mnli` · Gradio · Anthropic Claude API
- **Python:** 3.11+ · Windows (PowerShell) · venv at `.venv\`

## 2. Windows Runtime Rules (non-negotiable — always enforce)
- `num_workers=0` in **every** DataLoader call. Windows multiprocessing kills training.
- No `albumentations` / `albucore` — C++ compiler dependency breaks on Windows.
- Use `pathlib.Path` everywhere. Never `os.path` string concatenation.
- Activate venv before any pip/python: `.venv\Scripts\activate`

## 3. Ponytail Ladder (enforce before writing any new code)
- **L0:** Skip it — do we actually need this?
- **L1:** stdlib only
- **L2:** platform-native (transformers pipeline, gradio blocks)
- **L3:** installed dep (anthropic SDK)
- **L4:** one-liner custom
- **L5:** minimum custom class

Never jump to L5 if L2 solves it.

## 4. Claude's Role
Claude API is the **SYNTHESIZER / ADVISOR** only.
- Claude is **NEVER** a classifier or retriever.
- Claude receives only: ambiguous prediction outputs + label set → returns label critique + suggestions.
- Max **300 tokens** per call. Model: `claude-sonnet-4-6`.

## 5. Code Style
- Type hints on all function signatures.
- Docstrings: one-line summary + Args + Returns.
- No `print()` in production paths — use the `logging` module.
- All Claude API calls wrapped in `try/except` with graceful degradation.

## 6. Test Rules
- `pytest` only, no `unittest`.
- Mock all external API calls (HF Inference + Claude API).
- Coverage target: **80%** on `src/`.

## 7. GitHub Repo Contract
- README must have: badges, About, Demo GIF placeholder, Skills Demonstrated table,
  Installation, Usage, Architecture diagram link, Interview Q&A link.
- Every commit: conventional format (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
