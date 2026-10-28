"""Standalone inference + performance test for an exported MNIST ViT checkpoint.

Usage:
    vit-infer                                     # accuracy + benchmark on the auto device
    vit-infer --device cpu --device mps           # compare devices
    vit-infer --out results/bench.json            # also write results as JSON
    vit-infer --dump-samples samples              # write 10 MNIST test digits as PNGs
    vit-infer --image samples/*.png               # predict image files (white digit on black)
    vit-infer --image photo.png --invert          # dark digit on light background
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime

import torch
import torchvision.transforms as transforms

from vit import data
from vit.model import load_checkpoint, pick_device


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def accuracy(model, loader, device) -> tuple[float, float]:
    """Return (accuracy %, wall seconds incl. data loading)."""
    correct = total = 0
    sync(device)
    t0 = time.perf_counter()
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    sync(device)
    return 100.0 * correct / total, time.perf_counter() - t0


@torch.inference_mode()
def latency(model, x, device, warmup: int = 20, iters: int = 200) -> list[float]:
    """Per-call forward latency in ms for input batch x."""
    x = x.to(device)
    for _ in range(warmup):
        model(x)
    sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize(times_ms: list[float], batch_size: int) -> dict:
    p50 = statistics.median(times_ms)
    return {
        "batch_size": batch_size,
        "p50_ms": p50,
        "p95_ms": percentile(times_ms, 95),
        "ms_per_img": p50 / batch_size,
        "img_per_s": batch_size / p50 * 1000,
    }


@torch.inference_mode()
def predict_image(model, path: str, device, invert: bool = False) -> tuple[int, float]:
    from PIL import Image, ImageOps

    img = Image.open(path).convert("L").resize((28, 28))
    if invert:
        # MNIST is a white digit on black; invert dark-on-light images (photos, drawings)
        img = ImageOps.invert(img)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    probs = model(x).softmax(1)[0].cpu()
    top = int(probs.argmax())
    return top, float(probs[top])


def dump_samples(root: str, out_dir: str, count: int) -> None:
    """Write the first `count` MNIST test images as PNGs named <index>_label<digit>.png."""
    os.makedirs(out_dir, exist_ok=True)
    testset = data.mnist(train=False, root=root, with_transform=False)
    for i in range(count):
        img, label = testset[i]
        img.save(os.path.join(out_dir, f"{i:03d}_label{label}.png"))
    print(f"wrote {count} images to {out_dir}/")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/vit_mnist.pt")
    ap.add_argument("--device", action="append", help="cpu, mps, cuda or auto (repeatable)")
    ap.add_argument("--batch-sizes", default="1,64,256,1024")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--data", default=data.DEFAULT_DATA_DIR)
    ap.add_argument("--out", help="write benchmark results to this JSON file")
    ap.add_argument("--image", nargs="+", help="predict one or more image files and exit")
    ap.add_argument("--invert", action="store_true", help="invert --image (dark digit on light bg)")
    ap.add_argument(
        "--dump-samples", metavar="DIR", help="write MNIST test images as PNGs and exit"
    )
    ap.add_argument("--count", type=int, default=10, help="number of images for --dump-samples")
    args = ap.parse_args(argv)

    if args.dump_samples:
        dump_samples(args.data, args.dump_samples, args.count)
        return

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}\ntrain one with: vit-train")

    devices = [pick_device(d) for d in (args.device or ["auto"])]

    if args.image:
        missing = [p for p in args.image if not os.path.isfile(p)]
        if missing:
            sys.exit(
                f"image not found: {', '.join(missing)}\n"
                "create test images with: vit-infer --dump-samples samples"
            )
        model, _ = load_checkpoint(args.checkpoint, devices[0])
        for path in args.image:
            pred, conf = predict_image(model, path, devices[0], invert=args.invert)
            print(f"{path}: pred={pred} confidence={conf:.3f}")
        return

    testset = data.mnist(train=False, root=args.data)
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    sample = torch.stack([testset[i][0] for i in range(max(batch_sizes))])

    report = {
        "checkpoint": args.checkpoint,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
        "iters": args.iters,
        "devices": [],
    }
    for device in devices:
        model, ckpt = load_checkpoint(args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"\n=== {device} | {args.checkpoint} | params={n_params:,} | torch {torch.__version__}"
        )
        if ckpt.get("git_sha"):
            print(f"checkpoint: git {ckpt['git_sha']}, created {ckpt.get('created_at')}")

        acc, secs = accuracy(model, loader, device)
        print(
            f"test accuracy: {acc:.2f}%  ({len(testset)} images in {secs:.2f}s, incl. data loading)"
        )

        rows = [
            summarize(latency(model, sample[:bs], device, iters=args.iters), bs)
            for bs in batch_sizes
        ]
        print(f"{'batch':>6} {'p50 ms':>9} {'p95 ms':>9} {'ms/img':>9} {'img/s':>10}")
        for r in rows:
            print(
                f"{r['batch_size']:>6} {r['p50_ms']:>9.3f} {r['p95_ms']:>9.3f} "
                f"{r['ms_per_img']:>9.4f} {r['img_per_s']:>10,.0f}"
            )
        report["devices"].append(
            {
                "device": str(device),
                "params": n_params,
                "test_accuracy": acc,
                "checkpoint_git_sha": ckpt.get("git_sha"),
                "latency": rows,
            }
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
