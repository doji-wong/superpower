# ADR-0001: Use DSPy for Skill Description Optimization

## Status
Accepted

## Date
2026-08-31

## Context
Skill descriptions in this repository serve a dual purpose: they describe the skill to users, and they act as the primary routing mechanism for the agent's Tier-2 lexical evaluator (`run-evals.js`). Historically, writing descriptions that successfully hit the vocabulary required for high rank-1 routing accuracy across varied user prompts was a manual, trial-and-error process. We reached an 86% rank-1 baseline but struggled to improve it further without descriptions drifting toward each other (causing collisions).

We needed a systematic way to optimize these descriptions against the existing evaluation cases (`evals/cases/*.json`) to maximize the rank-1 trigger rate while minimizing pairwise collision.

## Decision
Use DSPy (specifically `scripts/dspy_optimizer.py` and `scripts/optimize_descriptions.py`) to systematically evaluate and optimize skill descriptions using an LLM (Gemini via the `agy` CLI).

## Alternatives Considered

### Manual Iteration
- Pros: No additional tooling required; fully human-authored.
- Cons: Extremely time-consuming; humans are bad at intuitively balancing TF-IDF weights across dozens of skills to avoid collisions. Reached a ceiling at ~86% accuracy.
- Rejected: Does not scale well as the skill catalog grows.

### Embeddings-based Routing
- Pros: Captures semantic meaning better than TF-IDF.
- Cons: Requires an embeddings model at runtime, complicating the lightweight, zero-dependency Node validation loop (`run-evals.js`).
- Rejected: We want to preserve the deterministic, zero-cost Tier 2 CI pipeline.

## Consequences
- **Improved Accuracy:** Tier-2 routing accuracy improved to 99% rank-1, allowing us to raise the CI floor (`--min-rank1`) from 80% to 95%.
- **Tooling Footprint:** Added `scripts/dspy_optimizer.py` and `scripts/optimize_descriptions.py` to the repository.
- **Workflow:** When adding new skills or triggers, developers can now run the DSPy optimizer to suggest mathematically optimal descriptions instead of guessing.
