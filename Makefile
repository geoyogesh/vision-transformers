.PHONY: help setup hooks test lint format train bench samples diagram notebook clean

CHECKPOINT ?= checkpoints/vit_mnist.pt
TRAIN_ARGS ?=
BENCH_ARGS ?= --device cpu --device mps

help:
	@echo "setup     install locked deps into .venv (uv sync)"
	@echo "hooks     install git pre-commit hooks (ruff + nbstripout)"
	@echo "test      run the test suite"
	@echo "lint      ruff check + format check"
	@echo "format    auto-fix lint and format code"
	@echo "train     train and export $(CHECKPOINT)  [TRAIN_ARGS=...]"
	@echo "bench     accuracy, confusion matrix + latency -> results/bench.json, docs/confusion_matrix.png"
	@echo "samples   write 10 MNIST test digits to samples/"
	@echo "diagram   render docs/architecture.png (needs: brew install graphviz)"
	@echo "notebook  start Jupyter Lab"
	@echo "clean     remove caches"

setup:
	uv sync

hooks: setup
	uv run pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

train:
	uv run vit-train --out $(CHECKPOINT) $(TRAIN_ARGS)

bench:
	uv run vit-infer --checkpoint $(CHECKPOINT) --out results/bench.json \
		--confusion-png docs/confusion_matrix.png $(BENCH_ARGS)

samples:
	uv run vit-infer --dump-samples samples

diagram:
	uv run python scripts/render_architecture.py

notebook:
	uv run jupyter lab notebooks/

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
