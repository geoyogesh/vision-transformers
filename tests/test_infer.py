"""Smoke tests for the inference CLI that need no dataset download."""

import subprocess
import sys

import pytest
import torch
from PIL import Image

from vit import VisionTransformer, ViTConfig, save_checkpoint
from vit.infer import latency, main, predict_image, summarize


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(VisionTransformer(ViTConfig()), str(path))
    return path


@pytest.fixture
def digit_png(tmp_path):
    path = tmp_path / "digit.png"
    Image.new("L", (28, 28), 0).save(path)
    return path


def test_predict_image(checkpoint, digit_png):
    from vit.model import load_checkpoint

    model, _ = load_checkpoint(str(checkpoint))
    pred, conf = predict_image(model, str(digit_png), torch.device("cpu"))
    assert 0 <= pred <= 9
    assert 0.0 < conf <= 1.0


def test_latency_summary():
    model = VisionTransformer(ViTConfig()).eval()
    times = latency(model, torch.rand(4, 1, 28, 28), torch.device("cpu"), warmup=2, iters=5)
    row = summarize(times, 4)
    assert len(times) == 5
    assert set(row) == {"batch_size", "p50_ms", "p95_ms", "ms_per_img", "img_per_s"}
    assert row["p50_ms"] > 0 and row["img_per_s"] > 0
    assert row["p95_ms"] >= row["p50_ms"]


def test_cli_predicts_images(checkpoint, digit_png, capsys):
    main(["--checkpoint", str(checkpoint), "--device", "cpu", "--image", str(digit_png)])
    assert "pred=" in capsys.readouterr().out


def test_cli_missing_image_exits_cleanly(checkpoint, tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--checkpoint", str(checkpoint), "--image", str(tmp_path / "nope.png")])
    assert "image not found" in str(exc.value)


def test_cli_missing_checkpoint_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--checkpoint", str(tmp_path / "nope.pt")])
    assert "checkpoint not found" in str(exc.value)


def test_entry_point_installed():
    out = subprocess.run(
        [sys.executable, "-m", "vit.infer", "--help"], capture_output=True, text=True, check=True
    )
    assert "--dump-samples" in out.stdout
