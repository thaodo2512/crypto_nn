# Repository Guidelines

## Project Structure & Module Organization
- `src/crypto_nn/` — core library code (models, data, utils).
- `scripts/` — runnable entry points (e.g., `train.py`, `eval.py`).
- `tests/` — unit/integration tests mirroring `src` layout.
- `notebooks/` — exploration; clear outputs before commit.
- `docs/` — short design notes/READMEs.
- `.env.example` — environment variable template for local runs.

## Build, Test, and Development Commands
- Setup venv: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.txt -r requirements-dev.txt`.
- Run tests: `pytest -q`.
- Coverage: `pytest --cov=crypto_nn --cov-report=term-missing`.
- Lint: `ruff check src tests`.
- Format: `black src tests`.
- Type check: `mypy src`.
- Run locally: `python -m crypto_nn.cli` or `python scripts/train.py`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indent, max line length 100.
- Names: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Require type hints and docstrings; prefer small, cohesive modules.
- Tools: Black (format), Ruff (lint/imports), Mypy (types, strict where practical).

## Testing Guidelines
- Framework: `pytest`; tests in `tests/` named `test_*.py` matching module paths.
- Write fast, deterministic tests; seed randomness where relevant.
- Target ≥85% statement coverage; include edge and failure paths.

## Commit & Pull Request Guidelines
- Conventional Commits (e.g., `feat(model): add attention layer`, `fix(train): handle NaNs`).
- PRs include: clear description, rationale, linked issues, before/after notes, and tests.
- CI gate: format, lint, type-check, tests, and coverage threshold must pass.

## Security & Configuration Tips
- Never commit secrets; use environment variables. Commit `.env.example`, not `.env`.
- Avoid large binaries; prefer small, reproducible datasets or downloads in setup scripts.

## Agent-Specific Instructions
- Keep patches minimal and focused; preserve layout under `src/crypto_nn` and `tests/`.
- Update/add tests when behavior changes; do not lower coverage without approval.
- Document new commands in this file or `docs/`.

