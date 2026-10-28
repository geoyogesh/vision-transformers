"""Train the MNIST ViT and export a checkpoint with provenance metadata.

Usage:
    vit-train                                   # 5 epochs -> checkpoints/vit_mnist.pt
    vit-train --epochs 10 --lr 1e-3 --out checkpoints/run2.pt
"""

import argparse
import os
import subprocess
import time
from datetime import UTC, datetime

import torch
import torch.nn as nn

from vit import data
from vit.config import TrainConfig, ViTConfig
from vit.model import VisionTransformer, pick_device, save_checkpoint


def git_sha() -> str | None:
    """Commit SHA of the working tree, with '-dirty' if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL
        ).returncode
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance(device: torch.device) -> dict:
    """Plain-type metadata (must stay loadable with torch.load(weights_only=True))."""
    return {
        "git_sha": git_sha(),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    loss_sum = correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


def main(argv: list[str] | None = None) -> None:
    defaults = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--epochs", type=int, default=defaults.epochs)
    ap.add_argument("--batch-size", type=int, default=defaults.batch_size)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data", default=data.DEFAULT_DATA_DIR)
    ap.add_argument("--out", default="checkpoints/vit_mnist.pt")
    args = ap.parse_args(argv)

    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    torch.manual_seed(tcfg.seed)
    device = pick_device(args.device)

    train_loader = data.loader(train=True, batch_size=tcfg.batch_size, root=args.data)
    val_loader = data.loader(train=False, batch_size=256, root=args.data)

    model = VisionTransformer(ViTConfig()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    history = []
    for epoch in range(tcfg.epochs):
        t0 = time.perf_counter()
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        history.append(
            {"epoch": epoch + 1, "loss": loss, "train_acc": train_acc, "val_acc": val_acc}
        )
        print(
            f"epoch {epoch + 1}: loss={loss:.4f} train_acc={train_acc:.2f}% "
            f"val_acc={val_acc:.2f}% ({time.perf_counter() - t0:.1f}s)"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_checkpoint(
        model.cpu(),
        args.out,
        val_acc=history[-1]["val_acc"],
        history=history,
        train_config=tcfg.to_dict(),
        **provenance(device),
    )
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
