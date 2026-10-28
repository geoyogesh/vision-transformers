"""The tutorial notebook keeps its own inline model classes; make sure they stay
weight-compatible and numerically identical to the vit package."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from vit import VisionTransformer, ViTConfig

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_vit_from_scratch.ipynb"
MARKERS = (
    "patch_size = ",
    "class PatchEmbedding",
    "class TransformerArchitecture",
    "class VisionTransformer",
)


def notebook_namespace() -> dict:
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    ns = {"torch": torch, "nn": nn}
    for marker in MARKERS:
        src = next(
            "".join(c["source"])
            for c in cells
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        )
        exec(compile(src, f"{NOTEBOOK.name}:{marker}", "exec"), ns)
    return ns


def test_notebook_hyperparameters_match_config():
    ns = notebook_namespace()
    cfg = ViTConfig()
    for field in (
        "img_size",
        "patch_size",
        "num_channels",
        "num_heads",
        "embed_dim",
        "mlp_dim",
        "transformer_units",
    ):
        assert ns[field] == getattr(cfg, field), field


def test_notebook_weights_load_into_package_model():
    ns = notebook_namespace()
    torch.manual_seed(0)
    nb_model = ns["VisionTransformer"]().eval()
    pkg_model = VisionTransformer(ViTConfig()).eval()
    pkg_model.load_state_dict(nb_model.state_dict(), strict=True)

    x = torch.rand(8, 1, 28, 28)
    with torch.no_grad():
        torch.testing.assert_close(pkg_model(x), nb_model(x), rtol=1e-5, atol=1e-6)
