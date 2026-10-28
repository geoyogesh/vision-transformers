import pytest
import torch

from vit import VisionTransformer, ViTConfig, load_checkpoint, save_checkpoint


@pytest.fixture
def model():
    torch.manual_seed(0)
    return VisionTransformer(ViTConfig()).eval()


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_output_shape(model, batch_size):
    out = model(torch.rand(batch_size, 1, 28, 28))
    assert out.shape == (batch_size, 10)


def test_param_count_is_stable(model):
    # Guards against accidental architecture changes; update deliberately if the config changes.
    assert sum(p.numel() for p in model.parameters()) == 2986


def test_config_drives_architecture():
    cfg = ViTConfig(embed_dim=32, num_heads=4, mlp_dim=64, transformer_units=3)
    m = VisionTransformer(cfg)
    assert len(m.transformer_layers) == 3
    assert m.pos_embed.shape == (1, cfg.num_patches + 1, 32)
    assert m(torch.rand(2, 1, 28, 28)).shape == (2, 10)


def test_checkpoint_round_trip(model, tmp_path):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, str(path), val_acc=12.5, git_sha="abc123")
    loaded, ckpt = load_checkpoint(str(path))
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
    assert ckpt["val_acc"] == 12.5
    assert ckpt["git_sha"] == "abc123"
    assert ckpt["config"] == ViTConfig().to_dict()


def test_training_provenance_is_weights_only_loadable(model, tmp_path):
    from vit.train import provenance

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, str(path), **provenance(torch.device("cpu")))
    _, ckpt = load_checkpoint(str(path))  # uses weights_only=True
    assert ckpt["torch_version"] == str(torch.__version__)
    assert ckpt["created_at"]


def test_overfits_single_batch():
    """If the model + loss + optimizer are wired correctly, one fixed batch is memorized."""
    torch.manual_seed(0)
    m = VisionTransformer(ViTConfig(embed_dim=32, mlp_dim=64))
    x, y = torch.rand(16, 1, 28, 28), torch.arange(16) % 10
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert (m(x).argmax(1) == y).all()
