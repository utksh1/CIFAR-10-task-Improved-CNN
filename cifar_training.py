#!/usr/bin/env python3
"""
CIFAR-10 small-data experiments: baseline vs improved CNNs.

Trains on a fixed random subset (default 10,000 images) of CIFAR-10 and
compares three recipes:

  baseline     3-layer SimpleCNN, no BatchNorm, no augmentation (the original task)
  improved_v1  SimpleCNN + BatchNorm + flip/crop augmentation (the original "improved")
  improved_v2  ResNet-style CNN + full recipe: BatchNorm, flip/crop/Cutout,
               AdamW with decoupled weight decay (excluded for norm/bias params),
               warmup + cosine LR schedule, label smoothing, best-checkpoint
               selection

Fixes over the original script
------------------------------
* Subset indices are drawn once from a dedicated ``torch.Generator``, so every
  configuration trains on the SAME subset. (The original consumed global RNG
  state between dataloader calls, silently training the baseline and the
  improved model on different 10k samples.)
* Per-run seeding (python / numpy / torch, DataLoader shuffle generator and
  worker seeds) makes runs reproducible.
* ``x.view(x.size(0), -1)`` replaced by ``torch.flatten(x, 1)``.
* Per-channel CIFAR-10 normalization statistics instead of (0.5, 0.5, 0.5).
* Training curves are exported to CSV, the best checkpoint is saved, and each
  run gets per-class accuracy, a confusion matrix and a "most confidently
  wrong" misclassification gallery.

Usage
-----
  python cifar_training.py                                  # all 3 configs
  python cifar_training.py --configs baseline improved_v2   # subset of configs
  python cifar_training.py --epochs 1 --out-dir outputs_calib   # quick check
  python cifar_training.py --plot-only                      # rebuild plots from CSVs
  python cifar_training.py --force --configs improved_v2    # retrain from scratch

Interrupted runs resume automatically: a `checkpoint.pt` (model, optimizer, scheduler,
history, best-so-far, shuffle state) is written atomically after every epoch, and the
next invocation of the same command continues from it. Completed runs are not
retrained unless `--force` is given.

Outputs (per config, under --out-dir/<config>/):
  history.csv, summary.json, best.pt, per_class.png, confusion.png, failures.png
  plus comparison.png / summary.json at the out-dir root and
  plots/comparison_plot.png for backward compatibility with the old README.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Use a non-interactive backend when running as a plain script (headless box);
# keep whatever backend a Jupyter kernel already configured otherwise.
if "ipykernel" not in sys.modules and "IPython" not in sys.modules:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants & experiment configurations
# ---------------------------------------------------------------------------

SEED = 42

# Per-channel CIFAR-10 statistics (slightly better than the 0.5/0.5 the
# original script used).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

CONFIGS = {
    "baseline": dict(
        arch="simple",
        batchnorm=False,
        augment="none",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        label_smoothing=0.0,
        scheduler="none",
        epochs=30,
        note="original task setup (no BN, no augmentation), now on a fixed subset",
    ),
    "improved_v1": dict(
        arch="simple",
        batchnorm=True,
        augment="basic",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        label_smoothing=0.0,
        scheduler="none",
        epochs=30,
        note="original 'improved' setup (BN + flip/crop), bug-fixed comparison",
    ),
    "improved_v2": dict(
        arch="resnet",
        batchnorm=True,
        augment="full",
        optimizer="adamw",
        lr=1e-3,
        weight_decay=5e-4,
        label_smoothing=0.1,
        scheduler="warm_cosine",
        epochs=100,
        note="ResNet-style CNN + Cutout + AdamW + warmup/cosine + label smoothing",
    ),
}

# ---------------------------------------------------------------------------
# Seeding / reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs (CUDA included when present)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp(choice: str, device: torch.device):
    """Return the autocast dtype for the requested AMP mode (None = disabled)."""
    if choice == "off":
        return None
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        if device.type == "cuda":
            return torch.float16
        print("[warn] fp16 autocast requires CUDA; disabling AMP")
        return None
    # auto: fp16 on CUDA, bfloat16 nowhere by default (CPU bf16 is only fast
    # on AVX512-BF16 machines; keep runs conservative).
    return torch.float16 if device.type == "cuda" else None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Cutout:
    """Random square masking (DeVries & Taylor, 2017).

    Applied on normalized tensors: 0 is the mean colour. A fixed-length
    window is centred on a uniformly random point, which may lie outside the
    image so the visible hole has variable size.
    """

    def __init__(self, length: int = 8, p: float = 1.0):
        self.length = length
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        h, w = img.shape[-2:]
        cy = random.uniform(0, h)
        cx = random.uniform(0, w)
        y1 = int(np.clip(cy - self.length / 2, 0, h))
        y2 = int(np.clip(cy + self.length / 2, 0, h))
        x1 = int(np.clip(cx - self.length / 2, 0, w))
        x2 = int(np.clip(cx + self.length / 2, 0, w))
        img = img.clone()
        img[:, y1:y2, x1:x2] = 0.0
        return img

    def __repr__(self) -> str:  # handy when printing pipelines
        return f"Cutout(length={self.length}, p={self.p})"


def build_transforms(augment: str = "none"):
    """Return (train_transform, test_transform) for an augmentation level."""
    base = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if augment == "none":
        train_tf = transforms.Compose(base)
    elif augment == "basic":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            + base
        )
    elif augment == "full":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            + base
            + [Cutout(length=8, p=1.0)]
        )
    else:
        raise ValueError(f"unknown augment level: {augment!r}")
    test_tf = transforms.Compose(base)
    return train_tf, test_tf


_subset_index_cache: dict = {}


def get_subset_indices(dataset_len: int, subset_size: int, seed: int):
    """Deterministically pick the subset shared by every configuration.

    Uses a dedicated generator so the global RNG state (advanced by training,
    shuffling, ...) can never change which images are selected. This fixes the
    original script's bug where the baseline and the improved model silently
    trained on different subsets.
    """
    key = (dataset_len, subset_size, seed)
    if key not in _subset_index_cache:
        g = torch.Generator().manual_seed(seed)
        _subset_index_cache[key] = (
            torch.randperm(dataset_len, generator=g)[:subset_size].tolist()
        )
    return _subset_index_cache[key]


def _make_worker_init_fn(seed: int):
    def init_fn(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed((seed + worker_id) % (2**32 - 1))

    return init_fn


def get_dataloaders(
    subset_size: int = 10000,
    batch_size: int = 128,
    augment: str = "none",
    data_dir: str = "./data",
    num_workers: int = 2,
    subset_seed: int = SEED,
    loader_seed: int = SEED,
    test_size: int = 0,
    test_batch_size: int = 256,
):
    """Build train/val loaders over the shared fixed subset of CIFAR-10."""
    train_tf, test_tf = build_transforms(augment)
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf
    )

    indices = get_subset_indices(len(trainset), subset_size, subset_seed)
    train_subset = Subset(trainset, indices)

    if test_size and test_size < len(testset):
        eval_testset = Subset(testset, list(range(test_size)))
    else:
        eval_testset = testset

    loader_kwargs = dict(
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        worker_init_fn=_make_worker_init_fn(loader_seed) if num_workers else None,
    )
    shuffle_gen = torch.Generator().manual_seed(loader_seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_gen,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        eval_testset, batch_size=test_batch_size, shuffle=False, **loader_kwargs
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """The original task's 3-conv CNN.

    Architecture is kept identical (channel widths, kernel sizes, pooling,
    FC sizes) so the baseline numbers remain comparable with the original
    repo; only ``view`` -> ``flatten`` and construction style changed.
    """

    def __init__(self, use_batchnorm: bool = False):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool2d(2, 2)]
            return nn.Sequential(*layers)

        self.conv_layer = nn.Sequential(block(3, 32), block(32, 64), block(64, 64))
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = torch.flatten(x, 1)
        return self.fc_layer(x)


class ResidualBlock(nn.Module):
    """Pre-activation-free (classic) residual block: conv-bn-relu-conv-bn + shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNetSmall(nn.Module):
    """Small CIFAR-style ResNet.

    Stem conv at 32x32, three stages (32x32 -> 16x16 -> 8x8) with two
    residual blocks each, global average pooling and a dropout head.
    ~170k parameters at width (16, 32, 64) - light enough for CPU training.
    """

    def __init__(
        self,
        widths=(16, 32, 64),
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
        num_classes: int = 10,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(),
        )
        stages = []
        in_ch = widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(ResidualBlock(in_ch, w, stride=stride))
            for _ in range(blocks_per_stage - 1):
                stages.append(ResidualBlock(w, w))
            in_ch = w
        self.body = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def build_model(arch: str, batchnorm: bool = False, width_mult: float = 1.0) -> nn.Module:
    if arch == "simple":
        return SimpleCNN(use_batchnorm=batchnorm)
    if arch == "resnet":
        widths = [max(8, int(round(c * width_mult))) for c in (16, 32, 64)]
        return ResNetSmall(widths=widths)
    raise ValueError(f"unknown arch: {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Optimization helpers
# ---------------------------------------------------------------------------


def param_groups_weight_decay(model: nn.Module, weight_decay: float):
    """AdamW parameter groups: no weight decay on 1-D params (norms, biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def make_lr_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_frac: float = 0.05):
    """Per-iteration linear warmup (5% of steps) then cosine decay to ~1% of lr."""
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.49 * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _NoopScaler:
    """GradScaler stand-in for non-fp16 training (bf16/off need no scaling)."""

    def scale(self, loss):
        return loss

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        pass


def autocast_ctx(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def save_checkpoint(path: str, *, model, optimizer, scheduler, history,
                    best_acc, best_epoch, best_state, epoch, shuffle_gen_state=None) -> None:
    """Atomically write a resumable training checkpoint (tmp file + os.replace)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict() if scheduler is not None else None,
        history=history,
        best_acc=best_acc,
        best_epoch=best_epoch,
        best_state=best_state,
        epoch=epoch,
        shuffle_gen_state=shuffle_gen_state,
    )
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    cfg: dict,
    device: torch.device,
    log_name: str = "train",
    checkpoint_path: str | None = None,
    resume: bool = True,
):
    """Train `model` for `epochs` epochs; track history and the best checkpoint.

    When `checkpoint_path` is given, a resumable checkpoint is saved after every
    epoch; if a checkpoint already exists (and `resume`), training continues from
    it. Returns (history rows, best_state_dict, best_stats dict).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.0)))

    if cfg.get("optimizer") == "adamw":
        optimizer = optim.AdamW(
            param_groups_weight_decay(model, float(cfg.get("weight_decay", 0.0))),
            lr=float(cfg["lr"]),
        )
    else:  # plain Adam: keeps baseline/v1 faithful to the original recipe
        optimizer = optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    scheduler = None
    if cfg.get("scheduler") == "warm_cosine":
        scheduler = make_lr_scheduler(optimizer, epochs, len(train_loader))

    amp_dtype = cfg.get("amp_dtype")
    if amp_dtype == torch.float16 and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = _NoopScaler()

    history: list = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    start_epoch = 0

    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        best_acc, best_epoch = ckpt["best_acc"], ckpt["best_epoch"]
        best_state = ckpt["best_state"]
        start_epoch = ckpt["epoch"]
        gen = getattr(train_loader, "generator", None)
        if gen is not None and ckpt.get("shuffle_gen_state") is not None:
            gen.set_state(ckpt["shuffle_gen_state"])
        print(
            f"[{log_name}] resuming from epoch {start_epoch + 1}/{epochs} "
            f"(best so far {best_acc:.2f}% @ {best_epoch})",
            flush=True,
        )

    t_run = time.time()

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        run_loss, seen, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device, amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            run_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        train_loss = run_loss / max(seen, 1)
        train_acc = 100.0 * correct / max(seen, 1)
        lr_now = float(optimizer.param_groups[0]["lr"])

        row = dict(
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            train_acc=round(train_acc, 2),
            val_loss=round(val_loss, 4),
            val_acc=round(val_acc, 2),
            lr=round(lr_now, 6),
            seconds=round(time.time() - t0, 1),
        )
        history.append(row)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

        print(
            f"[{log_name}] epoch {epoch + 1:>3}/{epochs} "
            f"train_loss {train_loss:.4f} train_acc {train_acc:5.2f}% "
            f"val_loss {val_loss:.4f} val_acc {val_acc:5.2f}% "
            f"(best {best_acc:.2f}% @ {best_epoch}) lr {lr_now:.2e}",
            flush=True,
        )

        if checkpoint_path:
            gen = getattr(train_loader, "generator", None)
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                best_acc=best_acc,
                best_epoch=best_epoch,
                best_state=best_state,
                epoch=epoch + 1,
                shuffle_gen_state=gen.get_state() if gen is not None else None,
            )

    best_stats = dict(
        best_val_acc=round(best_acc, 2),
        best_epoch=best_epoch,
        total_seconds=round(time.time() - t_run, 1),
    )
    return history, best_state, best_stats


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds: bool = False):
    """Evaluate; returns (loss, acc, per_class_acc[, preds, labels, confidences])."""
    model.eval()
    total_loss, seen, correct = 0.0, 0, 0
    class_correct = torch.zeros(len(CIFAR_CLASSES), dtype=torch.long)
    class_total = torch.zeros(len(CIFAR_CLASSES), dtype=torch.long)
    all_preds, all_labels, all_confs = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
        class_total += torch.bincount(labels, minlength=10).cpu()
        class_correct += torch.bincount(labels[preds == labels], minlength=10).cpu()
        if return_preds:
            confs = F.softmax(outputs.float(), dim=1).max(dim=1).values
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    val_loss = total_loss / max(seen, 1)
    val_acc = 100.0 * correct / max(seen, 1)
    per_class = (100.0 * class_correct / class_total.clamp(min=1)).tolist()
    if return_preds:
        return val_loss, val_acc, per_class, all_preds, all_labels, all_confs
    return val_loss, val_acc, per_class


def final_eval(
    model: nn.Module,
    best_state: dict,
    val_loader: DataLoader,
    out_dir: str,
    config_name: str,
    device: torch.device,
    max_failures: int = 12,
    save_plots: bool = True,
) -> dict:
    """Reload the best checkpoint and produce the full evaluation report."""
    model.load_state_dict(best_state)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, per_class, preds, labels, confs = evaluate(
        model, val_loader, criterion, device, return_preds=True
    )

    confusion = torch.bincount(
        torch.tensor(labels) * 10 + torch.tensor(preds), minlength=100
    ).reshape(10, 10)

    metrics = dict(
        val_loss=round(val_loss, 4),
        val_acc=round(val_acc, 2),
        per_class_acc=[round(a, 2) for a in per_class],
    )
    if save_plots:
        os.makedirs(out_dir, exist_ok=True)
        plot_per_class(per_class, os.path.join(out_dir, "per_class.png"), config_name)
        plot_confusion(confusion, os.path.join(out_dir, "confusion.png"), config_name)
        plot_failure_gallery(
            val_loader, preds, labels, confs,
            os.path.join(out_dir, "failures.png"), config_name, max_failures,
        )
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1)


def plot_per_class(per_class: list, out_path: str, config_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    y = np.arange(len(CIFAR_CLASSES))
    ax.barh(y, per_class, color="#4C72B0")
    ax.set_yticks(y, CIFAR_CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("accuracy (%)")
    ax.set_title(f"Per-class test accuracy - {config_name}")
    for i, v in enumerate(per_class):
        ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(confusion: torch.Tensor, out_path: str, config_name: str) -> None:
    cm = confusion.numpy().astype(float)
    cm_norm = 100.0 * cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10), CIFAR_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(10), CIFAR_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (% of true class) - {config_name}")
    for i in range(10):
        for j in range(10):
            color = "white" if cm_norm[i, j] > 50 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.85, label="%")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_failure_gallery(
    val_loader, preds, labels, confs, out_path, config_name, max_images: int = 12
) -> None:
    wrong = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
    if not wrong:
        print(f"[{config_name}] no misclassified samples to plot")
        return
    wrong.sort(key=lambda i: -confs[i])  # most confidently wrong first
    chosen = wrong[:max_images]
    want = set(chosen)
    imgs = {}
    base_idx = 0
    for images, _ in val_loader:
        for k in range(images.size(0)):
            if base_idx + k in want:
                imgs[base_idx + k] = images[k]
        base_idx += images.size(0)

    cols, rows = 4, math.ceil(len(chosen) / 4)
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.5 * rows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, chosen):
        ax.imshow(unnormalize(imgs[i]).permute(1, 2, 0).numpy())
        ax.set_title(
            f"true {CIFAR_CLASSES[labels[i]]}\npred {CIFAR_CLASSES[preds[i]]} ({confs[i]:.2f})",
            fontsize=7,
        )
    fig.suptitle(f"Most confidently wrong predictions - {config_name}", fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def comparison_plot(histories: dict, out_paths) -> None:
    """2x2 panel: train loss / val loss / train acc / val acc across configs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"baseline": "#888888", "improved_v1": "#4C72B0", "improved_v2": "#C44E52"}
    panels = [
        ("train_loss", "training loss", "lower is better"),
        ("val_loss", "validation loss", "lower is better"),
        ("train_acc", "training accuracy (%)", "higher is better"),
        ("val_acc", "validation accuracy (%)", "higher is better"),
    ]
    for ax, (key, title, _) in zip(axes.ravel(), panels):
        for name, rows in histories.items():
            xs = [r["epoch"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, label=name, color=colors.get(name), linewidth=1.8)
            if key == "val_acc":
                bi = max(range(len(rows)), key=lambda k: rows[k][key])
                ax.scatter([xs[bi]], [ys[bi]], color=colors.get(name), zorder=3, s=25)
                ax.annotate(
                    f"{ys[bi]:.1f}%", (xs[bi], ys[bi]),
                    textcoords="offset points", xytext=(0, 6), fontsize=8,
                    color=colors.get(name), ha="center",
                )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("CIFAR-10 10k-subset experiments: recipe comparison", fontsize=13)
    for p in out_paths:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        fig.savefig(p, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def save_history_csv(history: list, path: str) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def load_history_csv(path: str) -> list:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def run_experiment(
    config_name: str,
    epochs: int | None = None,
    subset_size: int = 10000,
    batch_size: int = 128,
    seed: int = SEED,
    data_dir: str = "./data",
    out_dir: str = "./outputs",
    num_workers: int = 2,
    device: torch.device | None = None,
    amp: str = "auto",
    test_size: int = 0,
    width_mult: float = 1.0,
    max_failures: int = 12,
    save_artifacts: bool = True,
    lr: float | None = None,
    resume: bool = True,
    force: bool = False,
):
    """Run one configuration end-to-end and save all artifacts.

    Behavior:
      * If a resumable checkpoint exists, training continues from it.
      * If the config already completed before (summary.json present, no
        checkpoint), existing artifacts are loaded instead of retraining
        (pass force=True to retrain).
      * If this invocation ends before `epochs` is reached (e.g. it was
        interrupted), the returned summary carries completed=False and no
        final artifacts are written; rerun the same command to resume.

    Returns dict(summary, history, best_state, arch, config, completed) so
    callers (e.g. the notebook) can keep plotting/evaluating without re-training.
    """
    device = device or get_device()
    cfg = dict(CONFIGS[config_name])
    if epochs is not None:
        cfg["epochs"] = epochs
    if lr is not None:
        cfg["lr"] = lr
    cfg["amp_dtype"] = resolve_amp(amp, device)

    cfg_dir = os.path.join(out_dir, config_name)
    checkpoint_path = os.path.join(cfg_dir, "checkpoint.pt")
    summary_path = os.path.join(cfg_dir, "summary.json")

    if force and os.path.isdir(cfg_dir):
        import shutil

        shutil.rmtree(cfg_dir)
        print(f"[{config_name}] --force: removed existing {cfg_dir}", flush=True)

    # Already completed on a previous invocation: load artifacts, don't retrain.
    if (
        resume
        and not force
        and os.path.exists(summary_path)
        and not os.path.exists(checkpoint_path)
    ):
        with open(summary_path) as f:
            summary = json.load(f)
        history = load_history_csv(os.path.join(cfg_dir, "history.csv"))
        ckpt = torch.load(os.path.join(cfg_dir, "best.pt"), weights_only=True)
        print(
            f"\n=== {config_name} ===\n"
            f"  already complete (best {summary['best_val_acc']:.2f}% @ epoch "
            f"{summary['best_epoch']}) - loading existing artifacts, skipping training\n"
            f"  use --force to retrain from scratch",
            flush=True,
        )
        return dict(summary=summary, history=history, best_state=ckpt["state_dict"],
                    arch=summary["arch"], config=cfg, completed=True)

    # Same subset for every config (subset_seed = seed); per-config stream for
    # shuffling/augmentation (stable crc32-derived offset).
    run_seed = (seed + zlib.crc32(config_name.encode())) % (2**31)
    set_seed(run_seed)

    train_loader, val_loader = get_dataloaders(
        subset_size=subset_size,
        batch_size=batch_size,
        augment=cfg["augment"],
        data_dir=data_dir,
        num_workers=num_workers,
        subset_seed=seed,
        loader_seed=run_seed,
        test_size=test_size,
    )

    model = build_model(cfg["arch"], cfg.get("batchnorm", True), width_mult)
    n_params = count_parameters(model)

    print(
        f"\n=== {config_name} ===\n"
        f"  arch={cfg['arch']}  params={n_params:,}  epochs={cfg['epochs']}  "
        f"augment={cfg['augment']}\n"
        f"  optimizer={cfg['optimizer']} (lr={cfg['lr']}, wd={cfg.get('weight_decay', 0)})  "
        f"label_smoothing={cfg.get('label_smoothing', 0)}  "
        f"scheduler={cfg.get('scheduler', 'none')}\n"
        f"  subset={subset_size} (seed {seed}, shared across configs)  "
        f"device={device}  amp={cfg['amp_dtype']}\n"
        f"  note: {cfg.get('note', '')}",
        flush=True,
    )

    history, best_state, best_stats = train_model(
        model, train_loader, val_loader, cfg["epochs"], cfg, device,
        log_name=config_name,
        checkpoint_path=checkpoint_path if save_artifacts else None,
        resume=resume,
    )

    completed = len(history) >= cfg["epochs"]

    if not completed:
        print(
            f"[{config_name}] incomplete ({len(history)}/{cfg['epochs']} epochs) - "
            f"rerun the same command to resume from checkpoint.pt",
            flush=True,
        )
        return dict(
            summary=dict(config=config_name, arch=cfg["arch"], epochs=cfg["epochs"],
                         epochs_done=len(history), completed=False,
                         best_val_acc=best_stats["best_val_acc"],
                         best_epoch=best_stats["best_epoch"]),
            history=history, best_state=best_state, arch=cfg["arch"],
            config=cfg, completed=False,
        )

    metrics = final_eval(
        model, best_state, val_loader,
        os.path.join(out_dir, config_name) if save_artifacts else out_dir,
        config_name, device, max_failures=max_failures, save_plots=save_artifacts,
    )

    summary = dict(
        config=config_name,
        arch=cfg["arch"],
        params=n_params,
        epochs=cfg["epochs"],
        subset_size=subset_size,
        seed=seed,
        best_val_acc=best_stats["best_val_acc"],
        best_epoch=best_stats["best_epoch"],
        final_train_acc=history[-1]["train_acc"] if history else None,
        final_train_loss=history[-1]["train_loss"] if history else None,
        reloaded_val_acc=metrics["val_acc"],
        per_class_acc=metrics["per_class_acc"],
        # Sum of per-epoch times from the history so interrupted+resumed runs
        # report the true total training time, not just the last invocation.
        train_seconds=round(sum(r.get("seconds", 0.0) for r in history), 1),
        optimizer=cfg["optimizer"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        label_smoothing=cfg.get("label_smoothing", 0.0),
        scheduler=cfg.get("scheduler", "none"),
        augment=cfg["augment"],
        device=str(device),
        amp=str(cfg["amp_dtype"]),
        torch_version=str(torch.__version__),
        note=cfg.get("note", ""),
        completed=True,
    )

    if save_artifacts:
        os.makedirs(cfg_dir, exist_ok=True)
        save_history_csv(history, os.path.join(cfg_dir, "history.csv"))
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        torch.save(
            {
                "state_dict": best_state,
                "config": {**cfg, "amp_dtype": str(cfg.get("amp_dtype"))},
                "history": history,
                "summary": {k: v for k, v in summary.items() if k != "per_class_acc"},
            },
            os.path.join(cfg_dir, "best.pt"),
        )
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)  # training finished; keep artifacts clean
        print(
            f"[{config_name}] artifacts saved to {cfg_dir}/ "
            f"(history.csv, summary.json, best.pt, per_class.png, confusion.png, failures.png)",
            flush=True,
        )

    return dict(summary=summary, history=history, best_state=best_state,
                arch=cfg["arch"], config=cfg, completed=True)


def print_summary_table(summaries: list) -> None:
    if not summaries:
        return
    header = f"{'config':<12}{'arch':<8}{'params':>9}{'epochs':>8}{'best acc':>10}{'best ep':>9}{'train acc':>11}{'minutes':>9}"
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['config']:<12}{s['arch']:<8}{s['params']:>9,}{s['epochs']:>8}"
            f"{s['best_val_acc']:>9.2f}%{s['best_epoch']:>9}"
            f"{(s['final_train_acc'] or 0):>10.2f}%{s['train_seconds'] / 60:>9.1f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS),
                   help="which configurations to run")
    p.add_argument("--epochs", type=int, default=None,
                   help="override epochs for every config (default: per-config preset)")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--subset-size", type=int, default=10000, help="training subset size")
    p.add_argument("--batch-size", type=int, default=128, help="training batch size")
    p.add_argument("--test-batch-size", type=int, default=256, help="evaluation batch size")
    p.add_argument("--test-size", type=int, default=0,
                   help="evaluate on only the first N test images (0 = full 10k test set)")
    p.add_argument("--seed", type=int, default=SEED, help="base seed (fixes the shared subset)")
    p.add_argument("--data-dir", default="./data", help="CIFAR-10 download/cache directory")
    p.add_argument("--out-dir", default="./outputs", help="artifact output directory")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--width-mult", type=float, default=1.0,
                   help="channel width multiplier for the ResNet-style model")
    p.add_argument("--amp", choices=("auto", "off", "bf16", "fp16"), default="auto",
                   help="mixed precision mode (auto: fp16 on CUDA, off on CPU)")
    p.add_argument("--max-failures", type=int, default=12,
                   help="images in the misclassification gallery")
    p.add_argument("--plot-only", action="store_true",
                   help="skip training; rebuild comparison plots from existing CSVs")
    p.add_argument("--smoke", action="store_true",
                   help="tiny fast run (subset 1000, 2 epochs) to verify the pipeline")
    p.add_argument("--force", action="store_true",
                   help="retrain selected configs from scratch (ignore completed runs)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = get_device()
    print(f"device: {device} | torch {torch.__version__} | CPUs visible: {os.cpu_count()}",
          flush=True)

    if args.smoke:
        args.subset_size, args.epochs, args.test_size = 1000, 2, 1000
        args.out_dir = "./outputs_smoke"
        args.max_failures = 8
        print("[smoke] subset=1000 epochs=2 test_size=1000 out_dir=outputs_smoke", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summaries, histories = [], {}

    if args.plot_only:
        for name in args.configs:
            path = os.path.join(args.out_dir, name, "history.csv")
            if os.path.exists(path):
                histories[name] = load_history_csv(path)
                with open(os.path.join(args.out_dir, name, "summary.json")) as f:
                    summaries.append(json.load(f))
            else:
                print(f"[plot-only] missing {path}, skipping {name}")
    else:
        for name in args.configs:
            result = run_experiment(
                name,
                epochs=args.epochs,
                subset_size=args.subset_size,
                batch_size=args.batch_size,
                seed=args.seed,
                data_dir=args.data_dir,
                out_dir=args.out_dir,
                num_workers=args.num_workers,
                device=device,
                amp=args.amp,
                test_size=args.test_size,
                width_mult=args.width_mult,
                max_failures=args.max_failures,
                lr=args.lr,
                force=args.force,
            )
            if result.get("completed", True):
                summaries.append(result["summary"])
                histories[name] = result["history"]

    if histories:
        comparison_plot(
            histories,
            [
                os.path.join(args.out_dir, "comparison.png"),
                os.path.join("plots", "comparison_plot.png"),
            ],
        )
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)
        print_summary_table(summaries)
        print(
            f"\ncomparison plot -> {args.out_dir}/comparison.png and plots/comparison_plot.png\n"
            f"run summaries   -> {args.out_dir}/summary.json",
            flush=True,
        )


if __name__ == "__main__":
    main()
