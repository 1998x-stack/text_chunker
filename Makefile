.PHONY: install test lint format benchmark experiment ci docker-build docker-run clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with all extras and dev dependencies
	bash scripts/install.sh

test: ## Run tests with coverage
	bash scripts/test.sh

lint: ## Run flake8 + mypy + isort check
	bash scripts/lint.sh

format: ## Format code with isort + black
	bash scripts/format.sh

benchmark: ## Benchmark all strategies on sample data
	bash scripts/benchmark.sh

experiment: ## Run ablation experiments
	bash scripts/experiment.sh

ci: ## Run full CI pipeline (lint + test + benchmark)
	bash scripts/ci.sh

docker-build: ## Build Docker image
	bash scripts/docker-build.sh

docker-run: ## Run chunker in Docker container
	docker-compose up

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .mypy_cache logs/ stats/ results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
