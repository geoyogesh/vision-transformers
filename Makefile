.PHONY: help setup hooks test lint format train bench samples notebook clean

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
	@echo "bench     accuracy + latency benchmark -> results/bench.json  [BENCH_ARGS=...]"
	@echo "samples   write 10 MNIST test digits to samples/"
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
	uv run vit-infer --checkpoint $(CHECKPOINT) --out results/bench.json $(BENCH_ARGS)

samples:
	uv run vit-infer --dump-samples samples

notebook:
	uv run jupyter lab notebooks/

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
