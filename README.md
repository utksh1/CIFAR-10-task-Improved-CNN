# CIFAR-10 Small-Data Experiments

A technical challenge: train a CNN on a small subset (10,000 samples) of CIFAR-10, then improve it as much as possible. This repo contains the training script, a narrative notebook, and the measured results of three recipes.

## Results (10k subset, full 10k test set, CPU-only, seed 42)

| config | model | params | recipe | epochs | best test acc | final train acc |
|---|---|---|---|---|---|---|
| `baseline` | SimpleCNN (3 conv) | 189k | plain Adam, no regularization | 30 | **64.28%** (ep 29) | 100.00% |
| `improved_v1` | SimpleCNN + BatchNorm | 189k | + flip/crop augmentation | 30 | **72.79%** (ep 30) | 77.47% |
| `improved_v2` | small ResNet-style CNN | 175k | + Cutout, AdamW + weight decay, warmup/cosine LR, label smoothing | 100 | **79.89%** (ep 99) | 90.79% |

Original repo numbers for reference: baseline 61.66%, improved 69.31% (10 epochs). The v2 recipe adds **+15.6 points** over the baseline and **+10.6** over the original improved setup.

Read the table with the last column in mind: the baseline memorizes the subset (100% train accuracy) while validation stalls — that is the small-data problem in one picture. Every recipe change in v1/v2 exists to shrink that gap.

## The bug that silently invalidated the original comparison

The original `get_dataloaders()` drew the subset with `torch.randperm` on the **global** RNG. After the baseline finished training, the RNG state had advanced, so the "improved" run silently trained on a **different** 10k subset — the two models were never compared on equal footing. The fix: subset indices are drawn once from a dedicated `torch.Generator` (`get_subset_indices`), cached, and shared by every configuration, no matter how much randomness each run consumed.

Other reproducibility fixes: per-run seeding of python/numpy/torch, a dedicated shuffle generator for the DataLoader (also saved/restored on resume), and worker-seed init for augmentation workers.

## What each improvement does (and why it helps with 10k images)

- **BatchNorm** — stabilizes gradients and lets the network tolerate a higher effective learning rate; with little data, activations drift more, so this pays off double.
- **Flip + random crop (pad 4)** — cheap label-preserving transformations; each epoch shows the model a slightly different view of the same 10k images, which directly fights memorization.
- **Cutout (8px)** — zeroes a random window per image (DeVries & Taylor, 2017); forces redundant features instead of one lucky patch. One of the highest-value tricks on CIFAR at this data scale.
- **Residual blocks + global average pooling** — deeper features without the optimization pain; GAP removes the giant flatten→FC layer (the biggest overfitting surface in the original model).
- **AdamW with decoupled weight decay** (excluded for norm/bias params) — proper L2 handling for Adam-family optimizers.
- **Warmup + cosine LR decay** — 5% linear warmup then cosine to ~1% of peak; the final low-LR phase typically buys the last 1-2 points.
- **Label smoothing 0.1** — softens over-confident targets, a mild but reliable regularizer.
- **Best-checkpoint selection** — the reported number is the best validation epoch, not the last one (the baseline's curve bounces near the end).

## Repo structure

```
cifar_training.py        # all experiments: models, training, eval, plots, CLI
cifar10_cnn_task.ipynb   # narrative walkthrough (imports the script, so they stay in sync)
outputs/                 # measured results: per-config history.csv, summary.json,
                         # best.pt, per_class.png, confusion.png, failures.png
plots/comparison_plot.png  # headline comparison figure (old README path, kept alive)
requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

python cifar_training.py                       # run all 3 configs (CPU: ~1h total)
python cifar_training.py --configs improved_v2 # one config
python cifar_training.py --smoke               # 2-epoch pipeline check (~1 min)
python cifar_training.py --plot-only           # rebuild plots from saved CSVs
python cifar_training.py --force --configs baseline  # retrain from scratch
```

Useful flags: `--epochs`, `--subset-size`, `--batch-size`, `--lr`, `--seed`,
`--width-mult` (widen the ResNet), `--amp {auto,off,bf16,fp16}`, `--num-workers`,
`--data-dir`, `--out-dir`. See `--help`.

**Interrupted runs resume automatically**: a checkpoint (model, optimizer, scheduler, history, shuffle state) is written atomically after every epoch; rerun the same command to continue. Completed runs are skipped (use `--force` to retrain).

The notebook runs the same three configs with lighter budgets (15/15/30 epochs) and shows the augmentation preview, per-class accuracy, confusion matrix and the "most confidently wrong" gallery. Full-budget numbers above come from the script defaults.

## Reproducibility notes

- Seed 42 everywhere; the subset, initialization, shuffling and augmentation are all deterministic given the same `--num-workers` (worker streams restart on resume, so a resumed run can differ from an uninterrupted one by a few tenths of a percent).
- The baseline/v1 SimpleCNN keeps the exact original layer sizes (189k params ≈ the original's 189k), so numbers stay comparable with the original repo.
- Everything was measured on a 2-core CPU box with PyTorch 2.14; a GPU run with `--amp auto` is several times faster and typically lands 1-2 points higher (cudnn nondeterminism aside).

## Where v2 still fails

Per-class accuracy (from `outputs/improved_v2/per_class.png`): cat 63% is the weakest class, with bird 72% and dog 74% close behind — the confusion matrix shows the classic cat↔dog and bird↔airplane confusion pairs. `outputs/improved_v2/failures.png` collects the most confidently wrong predictions, which are mostly low-contrast animals.
