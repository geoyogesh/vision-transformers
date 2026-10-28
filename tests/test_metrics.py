import torch

from vit.infer import confusion_matrix, per_class_metrics, save_confusion_png, top_confusions

# 3 classes, 6 samples:
#   true:  0 0 1 1 2 2
#   pred:  0 1 1 1 2 0
PREDS = torch.tensor([0, 1, 1, 1, 2, 0])
LABELS = torch.tensor([0, 0, 1, 1, 2, 2])


def test_confusion_matrix_counts():
    cm = confusion_matrix(PREDS, LABELS, 3)
    expected = torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert torch.equal(cm, expected)
    assert cm.sum() == len(LABELS)
    assert cm.diag().sum() == (PREDS == LABELS).sum()


def test_confusion_matrix_includes_unseen_classes():
    cm = confusion_matrix(torch.tensor([0]), torch.tensor([0]), 10)
    assert cm.shape == (10, 10)
    assert cm[0, 0] == 1 and cm.sum() == 1


def test_per_class_metrics():
    m = per_class_metrics(confusion_matrix(PREDS, LABELS, 3))
    # class 1: predicted 3 times, 2 correct; 2 true samples, both found
    assert abs(m[1]["precision"] - 200 / 3) < 1e-9
    assert m[1]["recall"] == 100.0
    assert abs(m[1]["f1"] - 80.0) < 1e-9
    assert [x["support"] for x in m] == [2, 2, 2]


def test_per_class_metrics_handles_empty_class():
    m = per_class_metrics(torch.zeros(3, 3, dtype=torch.long))
    assert all(x["precision"] == x["recall"] == x["f1"] == 0 for x in m)


def test_top_confusions_ignores_correct_predictions():
    cm = torch.tensor([[50, 3, 0], [0, 40, 7], [1, 0, 60]])
    top = top_confusions(cm, k=5)
    assert top[0] == {"true": 1, "pred": 2, "count": 7}
    assert top[1] == {"true": 0, "pred": 1, "count": 3}
    assert all(t["true"] != t["pred"] for t in top)
    assert len(top) == 3  # zero-count cells are dropped


def test_save_confusion_png(tmp_path):
    path = tmp_path / "cm.png"
    save_confusion_png(confusion_matrix(PREDS, LABELS, 3), str(path), "test")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_evaluate_matrix_agrees_with_accuracy():
    from torch.utils.data import DataLoader, TensorDataset

    from vit import VisionTransformer, ViTConfig
    from vit.infer import evaluate

    torch.manual_seed(0)
    images, labels = torch.rand(50, 1, 28, 28), torch.randint(0, 10, (50,))
    model = VisionTransformer(ViTConfig()).eval()
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)

    acc, secs, cm = evaluate(model, loader, torch.device("cpu"))
    with torch.no_grad():
        expected = 100.0 * (model(images).argmax(1) == labels).float().mean().item()
    assert cm.shape == (10, 10) and cm.sum() == 50
    assert torch.equal(cm.sum(1), torch.bincount(labels, minlength=10))
    assert abs(acc - expected) < 1e-6
    assert abs(acc - 100.0 * cm.diag().sum().item() / 50) < 1e-6
    assert secs > 0
