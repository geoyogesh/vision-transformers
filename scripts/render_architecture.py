"""Render the model's layer graph to docs/architecture.png (needs the Graphviz binary)."""

import os

from torchview import draw_graph

from vit import VisionTransformer, ViTConfig

OUT_DIR = "docs"


def main() -> None:
    cfg = ViTConfig()
    graph = draw_graph(
        VisionTransformer(cfg),
        input_size=(1, cfg.num_channels, cfg.img_size, cfg.img_size),
        expand_nested=True,
        depth=3,
        device="cpu",
        graph_name="VisionTransformer",
    )
    graph.visual_graph.render(os.path.join(OUT_DIR, "architecture"), format="png", cleanup=True)
    print(f"wrote {OUT_DIR}/architecture.png")


if __name__ == "__main__":
    main()
