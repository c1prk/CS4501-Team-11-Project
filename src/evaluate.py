"""
evaluate.py — CNN evaluation with test-time augmentation and per-method breakdown

Produces the numbers you'll need for the final report:
  - overall test accuracy, precision, recall, F1 per class
  - AUC-ROC (with probability outputs)
  - confusion matrix (as PNG and as printed numbers)
  - per-manipulation-method accuracy (the analysis slide 9 gestures at but doesn't show)
  - optional test-time augmentation (horizontal-flip average): small but consistent boost

Usage (from project root):
  python -m src.cnn_pipeline.evaluate
  python -m src.cnn_pipeline.evaluate --tta                  # enable TTA
  python -m src.cnn_pipeline.evaluate --checkpoint data/weights/cnn_last.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from dataset import (
    DeepfakeDataset, build_file_index, get_eval_transform,
)
from cnn_pipeline.model import DeepfakeCNN


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR    = os.path.join(PROJECT_ROOT, "data", "splits")
WEIGHTS_DIR   = os.path.join(PROJECT_ROOT, "data", "weights")
FIGURES_DIR   = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def category_from_path(path):
    """Extract manipulation method (or 'actors'/'youtube') from file path."""
    parts = path.replace('\\', '/').split('/')
    for i, p in enumerate(parts):
        if p in ('manipulated_frames', 'original_frames') and i + 1 < len(parts):
            return parts[i + 1]
    return 'unknown'


@torch.no_grad()
def infer(model, loader, device, tta=False):
    """Run inference; optionally average logits over original + h-flip."""
    model.eval()
    all_probs, all_labels, all_paths = [], [], []

    for imgs, labels, paths in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        if tta:
            logits_flip = model(torch.flip(imgs, dims=[-1]))
            logits = (logits + logits_flip) / 2
        probs = F.softmax(logits, dim=1)[:, 1]  # P(fake)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            all_paths)


def plot_confusion(cm, out_path, title, acc):
    plt.figure(figsize=(5.5, 4.5))
    cm_pct = cm.astype(float) / cm.sum() * 100
    annot = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
                       for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title(f"{title}\nTest Accuracy: {acc*100:.1f}%")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, auc_val, out_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(fpr, tpr, lw=2, label=f'{title} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC — {title}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=os.path.join(WEIGHTS_DIR, 'cnn_best.pt'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--tta', action='store_true',
                        help='Enable test-time augmentation (h-flip averaging).')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- load checkpoint ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DeepfakeCNN(
        backbone=ckpt.get('backbone', 'resnet18'),
        dropout=ckpt.get('dropout', 0.4),
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  backbone={ckpt.get('backbone', 'resnet18')}  "
          f"epoch={ckpt.get('epoch','?')}  val_acc={ckpt.get('val_acc',0):.4f}")

    # --- data ---
    index = build_file_index(PROCESSED_DIR)
    test_ds = DeepfakeDataset(
        split_file=os.path.join(SPLITS_DIR, "test.txt"),
        processed_dir=PROCESSED_DIR,
        transform=get_eval_transform(),
        file_index=index,
        return_path=True,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=(device.type == 'cuda'))

    # --- inference ---
    print(f"\nRunning inference (TTA={'on' if args.tta else 'off'})...")
    probs, labels, paths = infer(model, test_loader, device, tta=args.tta)
    preds = (probs >= args.threshold).astype(int)

    # --- overall metrics ---
    acc = accuracy_score(labels, preds)
    try:
        auc_val = roc_auc_score(labels, probs)
    except ValueError:
        auc_val = float('nan')
    cm = confusion_matrix(labels, preds)

    print(f"\n=== Overall Test Results ===")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"AUC-ROC:   {auc_val:.4f}")
    print(f"\nConfusion matrix:")
    print(f"                 Predicted Real   Predicted Fake")
    print(f"  Actual Real    {cm[0,0]:>14}  {cm[0,1]:>15}")
    print(f"  Actual Fake    {cm[1,0]:>14}  {cm[1,1]:>15}")
    print(f"\n{classification_report(labels, preds, target_names=['Real','Fake'], digits=4)}")

    # --- save figures ---
    title_suffix = f" {'(TTA)' if args.tta else ''}".strip()
    cm_path = os.path.join(FIGURES_DIR, f"cnn_confusion_matrix{'_tta' if args.tta else ''}.png")
    roc_path = os.path.join(FIGURES_DIR, f"cnn_roc{'_tta' if args.tta else ''}.png")
    plot_confusion(cm, cm_path, f"CNN ({ckpt.get('backbone','resnet18')}){title_suffix}", acc)
    plot_roc(labels, probs, auc_val, roc_path, f"CNN{title_suffix}")
    print(f"\nFigures:\n  {cm_path}\n  {roc_path}")

    # --- per-category breakdown ---
    print(f"\n=== Per-category breakdown ===")
    cats = np.array([category_from_path(p) for p in paths])
    rows = []
    for cat in sorted(set(cats)):
        mask = cats == cat
        n = int(mask.sum())
        cat_acc = accuracy_score(labels[mask], preds[mask])
        # For manipulated categories, acc == recall-on-fake. For real, acc == recall-on-real.
        is_real_cat = cat in ('actors', 'youtube')
        kind = 'real' if is_real_cat else 'fake'
        rows.append((cat, kind, n, cat_acc))
        print(f"  {cat:<22} ({kind})  n={n:<5}  acc={cat_acc*100:6.2f}%")

    # Save per-category to CSV for the report
    import csv
    per_cat_path = os.path.join(FIGURES_DIR, "cnn_per_category.csv")
    with open(per_cat_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'kind', 'n_samples', 'accuracy'])
        w.writerows(rows)
    print(f"\nPer-category CSV: {per_cat_path}")


if __name__ == "__main__":
    main()