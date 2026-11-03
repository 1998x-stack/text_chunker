# Development, tests and release gates

## Layer boundaries

- `core/`: canonical source mapping and contracts; no heavy NLP dependencies.
- `parsers/`: canonical Unicode text and provenance; must disclose unknown source offsets.
- `chunkers/`: candidate selection/splitting; no independent evaluation policy.
- `evaluation/`: check source mapping and budgets without trusting chunker output.
- `pipeline/`: orchestration and fail-closed result assembly.
- `export.py`: output only after validation.

Preserve the legacy package name `textchunker` and the CLI's prior keys/flags when extending.

## Local commands

```bash
python -m pip install -e ".[token,structure,dev]" numpy
bash scripts/lint.sh  # Python compileall + fatal Flake8 checks
bash scripts/test.sh  # pytest and >=60% statement coverage
bash scripts/ci.sh
```

The current CI intentionally gates fatal Python errors, not strict typing or full formatting. Do not describe it as a complete mypy/isort pass.

## PR dependency and release process

1. Implement one coherent feature on a branch based on the current `main`.
2. Add deterministic golden cases for source offset preservation, Unicode, table/code boundaries, token budgets and max_chunks behavior.
3. Record the CI run and coverage; distinguish mock model tests from live model E2E.
4. Open PR to `main`. Do not merge failed checks or skip human code review.
5. If a stacked PR is merged to an intermediate branch, it is **not** on main. Follow up with a separate promotion PR and confirm main's tree.
6. Update README, architecture, usage and changelog in the same feature PR; keep historical docs marked as historical.

## v0.4 acceptance checklist

- Structural Markdown/HTML blocks retain source hierarchy and original text or explicitly unknown mappings.
- Code fences and tables are preserved intact when within budget; overflow is reported/split deterministically with diagnostics.
- Hybrid strategy respects strict tokenizer budgets, including overlap and long-element fallbacks.
- Source span coverage and page provenance pass independently evaluated tests.
- CI tests include no network or live model requirements.
