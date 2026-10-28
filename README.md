# Vision Transformer from scratch (MNIST)

A minimal Vision Transformer (ViT) in PyTorch, trained on MNIST. The notebook walks through
the architecture step by step; the `vit` package trains, exports and benchmarks the same model.

## Architecture

```mermaid
flowchart TD
    A["Input image<br/>[B, 1, 28, 28]"] --> B["Patch embedding<br/>Conv2d 7x7, stride 7<br/>[B, 16, 4, 4]"]
    B --> C["Flatten + transpose<br/>16 patches x 16 dims<br/>[B, 16, 16]"]
    CLS(["CLS token<br/>[1, 1, 16]"]) --> D
    C --> D["Concat CLS<br/>[B, 17, 16]"]
    POS(["Positional embedding<br/>[1, 17, 16]"]) --> E
    D --> E(("+"))

    subgraph BLOCK["Transformer block x1 (pre-norm)"]
        direction TB
        F["LayerNorm"] --> G["Multi-head self-attention<br/>1 head"]
        G --> H(("+"))
        H --> I["LayerNorm"]
        I --> J["MLP<br/>Linear 16-16, GELU, Linear 16-16"]
        J --> K(("+"))
    end

    E --> F
    E -. residual .-> H
    H -. residual .-> K
    K --> L["Take CLS output<br/>[B, 16]"]
    L --> M["LayerNorm + Linear 16-10<br/>[B, 10] logits"]
```

Layer graph generated from the model with torchview (`make diagram`):

<img src="docs/architecture.png" alt="VisionTransformer layer graph with tensor shapes" width="320">

| Stage | Output shape |
|---|---|
| Input | `[B, 1, 28, 28]` |
| Patch embedding (Conv2d 7x7, stride 7) | `[B, 16, 4, 4]` |
| Flatten + transpose (16 patches x 16 dims) | `[B, 16, 16]` |
| Prepend CLS token + add positional embedding | `[B, 17, 16]` |
| Transformer block x1: `x + MHA(LN(x))`, `x + MLP(LN(x))` | `[B, 17, 16]` |
| CLS output -> LayerNorm -> Linear | `[B, 10]` |

Default config (`src/vit/config.py`): `embed_dim=16`, `num_heads=1`, `mlp_dim=16`,
`transformer_units=1`.

| Module | Parameters |
|---|---|
| Patch embedding (Conv2d) | 800 |
| CLS token | 16 |
| Positional embedding | 272 |
| Transformer block: LayerNorm x2 | 64 |
| Transformer block: self-attention | 1,088 |
| Transformer block: MLP | 544 |
| Classification head (LayerNorm + Linear) | 202 |
| **Total** | **2,986** |

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
scripts/                  pre-commit helper + architecture diagram renderer
docs/architecture.png     layer graph (make diagram)
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
make diagram              # docs/architecture.png
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

## Test results

`make test`: 16 passed in ~5.5s (CPU only, no dataset download).

| Test | What it checks | Result |
|---|---|---|
| `test_output_shape[1, 7, 64]` | Logits are `[B, 10]` for several batch sizes | Pass |
| `test_param_count_is_stable` | Exactly 2,986 parameters (catches accidental architecture changes) | Pass |
| `test_config_drives_architecture` | Non-default `ViTConfig` (embed 32, 4 heads, 3 layers) builds and runs | Pass |
| `test_checkpoint_round_trip` | Save + load gives bit-identical outputs and keeps metadata | Pass |
| `test_training_provenance_is_weights_only_loadable` | Training metadata loads with `torch.load(weights_only=True)` | Pass |
| `test_overfits_single_batch` | Model + loss + optimizer can memorize one batch (loss under 0.05) | Pass |
| `test_predict_image` | Image file prediction returns a digit and confidence | Pass |
| `test_latency_summary` | Benchmark summary has p50/p95/throughput fields | Pass |
| `test_cli_predicts_images` | `vit-infer --image` end to end | Pass |
| `test_cli_missing_image_exits_cleanly` | Missing image gives a clear error, not a traceback | Pass |
| `test_cli_missing_checkpoint_exits_cleanly` | Missing checkpoint gives a clear error | Pass |
| `test_entry_point_installed` | `python -m vit.infer --help` works | Pass |
| `test_notebook_hyperparameters_match_config` | Notebook globals equal `ViTConfig` defaults | Pass |
| `test_notebook_weights_load_into_package_model` | Notebook model weights load into the package model with identical outputs | Pass |

## Performance

Measured on Apple M3 Pro, macOS, torch 2.14.1. Raw numbers: `results/bench.json`
(regenerate with `make train && make bench`).

### Training (`make train` defaults: 5 epochs, Adam lr 0.01, batch 64, seed 0, MPS)

| Epoch | Train loss | Train acc | Test acc |
|---|---|---|---|
| 1 | 0.864 | 69.53% | 78.97% |
| 2 | 0.614 | 78.70% | 82.70% |
| 3 | 0.538 | 81.78% | 84.53% |
| 4 | 0.471 | 84.59% | 84.57% |
| 5 | 0.430 | 86.24% | 86.52% |

~5-6s per epoch, ~30s total. Checkpoint size ~20 KB.

### Inference

Test accuracy: **86.52%** (10,000 MNIST test images, identical on CPU and MPS).
Full test set: ~0.33s on CPU, ~0.28s on MPS, including data loading.

| Batch | CPU p50 | CPU p95 | CPU img/s | MPS p50 | MPS p95 | MPS img/s |
|---|---|---|---|---|---|---|
| 1 | 0.13 ms | 0.15 ms | ~7.6k | 0.50 ms | 0.59 ms | ~2.0k |
| 64 | 1.37 ms | 1.48 ms | ~47k | 0.55 ms | 0.58 ms | ~116k |
| 256 | 4.97 ms | 5.20 ms | ~52k | 0.48 ms | 0.53 ms | ~528k |
| 1024 | 19.1 ms | 19.6 ms | ~53k | 0.89 ms | 0.92 ms | ~1.15M |

- Single-image latency: CPU wins (~4x), since MPS dispatch overhead dominates a model this small.
- Batched throughput: MPS wins (~22x at batch 1024); CPU saturates around ~50k img/s.

## Known limitations

Training at `lr=0.01` is unstable: unseeded notebook runs have ranged from ~51% to ~90% test
accuracy. Likely improvements: AdamW 1e-3 + cosine schedule, `trunc_normal_(std=0.02)` init for
CLS/positional embeddings, input normalization, and a larger model (e.g. embed 64, 4 heads,
4-6 layers, mlp 128). A ViT this size on MNIST should reach ~97-98%.

## Contributing

Pre-commit runs ruff (lint + format) and strips notebook outputs on commit, so notebooks are
committed without outputs. Run `make test lint` before committing.
