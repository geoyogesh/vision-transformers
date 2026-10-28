# Vision Transformer from scratch (MNIST)

A minimal Vision Transformer (ViT) in PyTorch, trained on MNIST. The notebook walks through
the architecture step by step; the `vit` package trains, exports and benchmarks the same model.

## Architecture

| Stage | Shape |
|---|---|
| Input | `[B, 1, 28, 28]` |
| Patch embedding (Conv2d 7x7, stride 7) | `[B, 16, 4, 4]` |
| Flatten + transpose (16 patches x 16 dims) | `[B, 16, 16]` |
| Prepend CLS token + add positional embedding | `[B, 17, 16]` |
| Transformer block x1 (pre-norm: `x + MHA(LN(x))`, `x + MLP(LN(x))`) | `[B, 17, 16]` |
| CLS output -> LayerNorm -> Linear | `[B, 10]` |

Default config (`src/vit/config.py`): `embed_dim=16`, `num_heads=1`, `mlp_dim=16`,
`transformer_units=1`, 2,986 parameters.

## Setup

Requires [mise](https://mise.jdx.dev) (or Python 3.12) and [uv](https://docs.astral.sh/uv/).

```
mise install          # Python 3.12 from mise.toml
make setup            # uv sync: exact versions from uv.lock into .venv
make hooks            # once per clone: pre-commit hooks
```

The notebook's graph cell also needs the Graphviz binary: `brew install graphviz`.

Dependencies live in `pyproject.toml` (pinned) and `uv.lock` (full lock). Add one with
`uv add <pkg>` (or `uv add --group notebook <pkg>` / `uv add --dev <pkg>`); never `uv pip install`,
which bypasses the lockfile.

## Layout

```
pyproject.toml            deps, entry points, ruff + pytest config
uv.lock                   locked dependency versions (committed)
Makefile                  common tasks (make help)
src/vit/
  config.py               ViTConfig (model) + TrainConfig: single source of hyperparameters
  model.py                VisionTransformer, checkpoint save/load, device selection
  data.py                 MNIST datasets/loaders
  train.py                vit-train: train + export checkpoint with provenance
  infer.py                vit-infer: accuracy, latency benchmark, image prediction
notebooks/
  01_vit_from_scratch.ipynb   step-by-step tutorial + visualizations
tests/                    model, inference CLI, notebook/package parity
scripts/                  pre-commit helpers
results/bench.json        latest benchmark (committed)
checkpoints/ data/ samples/   generated, gitignored
```

The notebook intentionally keeps its own inline model classes as a from-scratch tutorial.
`tests/test_notebook_parity.py` checks its hyperparameters match `ViTConfig` and that its
weights load into `vit.VisionTransformer` with identical outputs, so the two cannot drift.

## Common tasks

```
make help
make test                 # pytest (~6s, CPU only, no dataset download)
make lint                 # ruff check + format check
make format               # auto-fix
make notebook             # jupyter lab notebooks/
make train                # -> checkpoints/vit_mnist.pt
make train TRAIN_ARGS="--epochs 10 --lr 1e-3" CHECKPOINT=checkpoints/run2.pt
make bench                # cpu + mps benchmark -> results/bench.json
make samples              # 10 MNIST test digits -> samples/
```

Or call the entry points directly (`uv run vit-train --help`, `uv run vit-infer --help`):

```
uv run vit-infer --device cpu --device mps --batch-sizes 1,32 --iters 500
uv run vit-infer --image samples/*.png
uv run vit-infer --image photo.png --invert     # dark digit on light background
```

Images are converted to grayscale and resized to 28x28. MNIST digits are white on black,
so use `--invert` for dark-on-light images. MNIST downloads to `./data` on first use.

## Reproducibility

- Training is seeded (`--seed`, default 0); the same seed + lockfile gives identical results.
- Each checkpoint stores the model config, training config, per-epoch history, final
  `val_acc`, git SHA (`-dirty` if uncommitted changes), torch version, device and timestamp.
- `make bench` writes `results/bench.json` with the machine, torch version, checkpoint SHA and
  per-device accuracy and latency. Commit it when results change.

## Results

From `results/bench.json` (Apple Silicon, torch 2.14.1, `make train` defaults, 5 epochs):

Test accuracy: 86.52%.

| Batch | CPU p50 | CPU img/s | MPS p50 | MPS img/s |
|---|---|---|---|---|
| 1 | 0.13 ms | ~7.6k | 0.50 ms | ~2.0k |
| 64 | 1.37 ms | ~47k | 0.55 ms | ~116k |
| 256 | 4.97 ms | ~52k | 0.49 ms | ~528k |
| 1024 | 19.1 ms | ~53k | 0.89 ms | ~1.15M |

- Single-image latency: CPU wins (~4x), since MPS dispatch overhead dominates a model this small.
- Batched throughput: MPS wins (~22x at batch 1024).

## Known limitations

Training at `lr=0.01` is unstable: unseeded notebook runs have ranged from ~51% to ~90% test
accuracy. Likely improvements: AdamW 1e-3 + cosine schedule, `trunc_normal_(std=0.02)` init for
CLS/positional embeddings, input normalization, and a larger model (e.g. embed 64, 4 heads,
4-6 layers, mlp 128). A ViT this size on MNIST should reach ~97-98%.

## Contributing

Pre-commit runs ruff (lint + format) and strips notebook outputs on commit, so notebooks are
committed without outputs. Run `make test lint` before committing.
