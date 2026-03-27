# Text-Chunker Project

## Architecture
- Factory + Registry pattern for pluggable chunking strategies
- 5 strategies: fixed, recursive, semantic, structure, llm
- DashScope API (DASHSCOPE_API_KEY) for LLM and embeddings
- Models: qwen-max (LLM), text-embedding-v3 (embeddings)
- Settings class centralizes API/model/logging/stats configuration

## Development
- Python >=3.9
- Install: `pip install -e ".[all,dev]"` or `make install`
- Test: `make test` or `pytest --cov=textchunker`
- Lint: `make lint` (flake8 + mypy + isort)
- Format: `make format` (isort + black)
- Always `source ~/.zshrc` before running commands

## Conventions
- Type hints on all public functions
- Loguru for logging (never print())
- snake_case everywhere
- Tests mock heavy deps (models, APIs)
- Custom exceptions: ChunkerError, ConfigError, ProviderError (in exceptions.py)

## Key Modules
- `settings.py` — Settings dataclass (API keys, model names, log config)
- `log_config.py` — Loguru setup (console + JSON file sinks)
- `stats.py` — StatsCollector singleton + @track_time / @count_calls decorators
- `factory.py` + `registry.py` — Strategy instantiation via @register decorator
- `chunkers/` — Strategy implementations (fixed, recursive, semantic, structure, llm_based)
- `providers/` — DashScope + HF backends (LLM and embedding)
- `experiments/` — Ablation studies and strategy comparison

## API Keys
- DASHSCOPE_API_KEY must be set in environment (loaded via `source ~/.zshrc`)
- Never hardcode API keys in source files

## Testing
- Stub sentence_transformers with dummy model in tests
- Mock DashScope/OpenAI API calls — never make real API calls in tests
- Use tmp_path fixture for file I/O tests
- Target: >60% coverage
