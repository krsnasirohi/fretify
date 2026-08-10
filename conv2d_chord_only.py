"""Single-task comparison architecture: the same joint time-frequency 2D
CNN backbone as conv2d_heads.py/conv2d_tab_only.py, but predicting chord
identity instead of pitch or tab -- a genuine third output, not an
auxiliary hint fed into another head.

Uses the plain/lead-sheet chord vocabulary (43 classes: 42 real chords +
NO_CHORD, see preprocessing/precompute_chord_annotations.py), not the
richer note-informed one (588 classes -- far too sparse for ~360
recordings). A chord label alone doesn't give you string/fret (the same
chord has many possible fingerings), so this model is not a tab
predictor -- it answers a different question ("what chord is this")
than the tab models do ("what fret, on what string, right now").

Needs its own Dataset (ChordGuitarDataset) and training loop rather than
reusing PrecomputedGuitarDataset/foundational_heads.py's infra directly:
the existing Dataset always returns a fixed (cqt, Y_pitch, Y_tab,
frame_mask) 4-tuple, a contract 6 other scripts depend on, so a
differently-shaped (cqt, Y_chord, frame_mask) dataset has to be new and
separate rather than bolted onto that one. Mirrors
PrecomputedGuitarDataset's sliding-window tiling (including the
tail-anchoring so no window needs padding except a clip shorter than
time_frames) and cross_validate_by_player_single_task's train/eval loop
structure, just sourcing chord labels from output/chord_annotations/
instead of pitch/tab from output/per_frame_annotations/.
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from foundational_heads import find_cqt_jams_pairs
from preprocessing.data_splits import grouped_split, player_id_from_stem, print_and_verify_split

CHORD_CACHE_DIR = "output/chord_annotations"

# ── 1. The Dataset ───────────────────────────────────────────────────

class ChordGuitarDataset(Dataset):
    """Loads pre-computed CQT tensors and pre-computed per-frame chord
    labels (see preprocessing/precompute_chord_annotations.py). Same
    sliding-window tiling as PrecomputedGuitarDataset in
    foundational_heads.py, just a different (cqt, Y_chord, frame_mask)
    return shape instead of that one's (cqt, Y_pitch, Y_tab, frame_mask).
    """
    def __init__(self, cqt_files, jams_files, time_frames=172, stride=129,
                 chord_cache_dir=CHORD_CACHE_DIR):
        self.cqt_files = cqt_files
        self.jams_files = jams_files
        self.time_frames = time_frames
        self.stride = stride
        self.chord_cache_dir = chord_cache_dir

        with open(os.path.join(chord_cache_dir, "_vocab.json")) as f:
            self.vocab = json.load(f)
        self.num_classes = len(self.vocab)
        self.no_chord_idx = self.vocab.index("NO_CHORD")

        self.windows = []
        for file_idx, cqt_path in enumerate(self.cqt_files):
            total_frames = np.load(cqt_path, mmap_mode='r').shape[0]
            if total_frames <= self.time_frames:
                starts = [0]
            else:
                starts = list(range(0, total_frames - self.time_frames + 1, self.stride))
                if starts[-1] + self.time_frames < total_frames:
                    starts.append(total_frames - self.time_frames)
            for start in starts:
                self.windows.append((file_idx, start))

    def __len__(self):
        return len(self.windows)

    def _load_chord_labels(self, jams_path, total_frames):
        stem = os.path.splitext(os.path.basename(jams_path))[0]
        cache_path = os.path.join(self.chord_cache_dir, f"{stem}.npz")
        cached = np.load(cache_path)
        return cached["Y_chord"][:total_frames]

    def __getitem__(self, idx):
        file_idx, start = self.windows[idx]
        cqt_full = np.load(self.cqt_files[file_idx], mmap_mode='r')
        total_frames = cqt_full.shape[0]
        end = min(start + self.time_frames, total_frames)
        cqt = np.array(cqt_full[start:end, :])

        Y_chord_full = self._load_chord_labels(self.jams_files[file_idx], total_frames)
        Y_chord = Y_chord_full[start:end]

        valid_frames = end - start
        frame_mask = np.zeros(self.time_frames, dtype=np.float32)
        frame_mask[:valid_frames] = 1.0

        if valid_frames < self.time_frames:
            pad_len = self.time_frames - valid_frames
            cqt = np.pad(cqt, ((0, pad_len), (0, 0)))
            Y_chord = np.pad(Y_chord, (0, pad_len), constant_values=self.no_chord_idx)

        cqt = np.expand_dims(cqt, axis=0)

        return torch.tensor(cqt, dtype=torch.float32), \
               torch.tensor(Y_chord, dtype=torch.long), \
               torch.tensor(frame_mask, dtype=torch.float32)


def compute_chord_class_counts(dataset: ChordGuitarDataset) -> np.ndarray:
    """Scans labels once to get per-class frame counts, shared by
    compute_chord_class_weights and the most-common-class baseline so
    neither has to re-scan the label cache."""
    counts = np.zeros(dataset.num_classes, dtype=np.int64)
    for jams_path, cqt_path in zip(dataset.jams_files, dataset.cqt_files):
        total_frames = np.load(cqt_path, mmap_mode='r').shape[0]
        Y_chord = dataset._load_chord_labels(jams_path, total_frames)
        counts += np.bincount(Y_chord, minlength=dataset.num_classes)
    return counts


def compute_chord_class_weights(counts: np.ndarray) -> torch.Tensor:
    """Same spirit as compute_class_weights in foundational_heads.py --
    counteracts class imbalance (mainly relevant for NO_CHORD, which is
    extremely rare, and the natural skew across the 42 real chords)."""
    freq = counts / counts.sum()
    weight = np.clip(1.0 / np.maximum(freq, 1e-4), 0.0, 20.0)
    weight = weight / weight.mean()
    return torch.tensor(weight, dtype=torch.float32)

# ── 2. The Model ─────────────────────────────────────────

class Conv2dChordOnlyModel(nn.Module):
    """Same joint time-frequency 2D CNN backbone as Conv2dGuitarModel, but
    with only chord_head -- forward() returns chord_probs, shape
    (batch, time_frames, num_classes).
    """
    def __init__(self, input_bins=252, num_classes=43, temporal_hidden=256,
                 latent_dim=128, dropout=0.4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 7), padding=(2, 3)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),              # freq 252 -> 126, time untouched
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=(1, 2)),              # freq 126 -> 63, time untouched
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),  # freq 63 -> 21, time untouched
        )

        self.freq_proj = nn.Sequential(
            nn.Linear(32 * 21, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal = nn.GRU(
            input_size=latent_dim,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        context_dim = temporal_hidden * 2
        self.context_dropout = nn.Dropout(dropout)

        self.chord_head = nn.Linear(context_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, _, time_frames, _ = x.shape

        feat = self.conv(x)                       # (batch, 32, time_frames, 21)
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size, time_frames, -1)
        latent = self.freq_proj(feat)

        context, _ = self.temporal(latent)
        context = self.context_dropout(context)

        chord_logits = self.chord_head(context)
        chord_probs = F.softmax(chord_logits, dim=-1)

        return chord_probs

# ── 3. Training & Evaluation ────────────────────────────────────────

def calculate_chord_loss(chord_probs, y_chord, frame_mask, class_weight):
    eps = 1e-7
    probs_permuted = chord_probs.permute(0, 2, 1).contiguous()  # (batch, classes, time)
    log_probs = torch.log(probs_permuted.clamp(min=eps))
    nll_elem = F.nll_loss(log_probs, y_chord, weight=class_weight, reduction='none')
    return (nll_elem * frame_mask).sum() / (frame_mask.sum() + eps)


@torch.no_grad()
def compute_chord_batch_metrics(chord_probs, y_chord, frame_mask, most_common_idx):
    mask = frame_mask.bool()
    pred = chord_probs.argmax(dim=-1)
    correct = ((pred == y_chord) & mask).sum().item()
    baseline_correct = ((y_chord == most_common_idx) & mask).sum().item()
    valid = mask.sum().item()
    return {"correct": correct, "baseline_correct": baseline_correct, "valid": valid}


def train_one_epoch_chord(model, loader, optimizer, device, class_weight, most_common_idx) -> dict:
    model.train()
    total_loss = 0.0
    totals = {"correct": 0, "baseline_correct": 0, "valid": 0}

    for x, y_chord, frame_mask in loader:
        x, y_chord, frame_mask = x.to(device), y_chord.to(device), frame_mask.to(device)

        optimizer.zero_grad()
        chord_probs = model(x)
        loss = calculate_chord_loss(chord_probs, y_chord, frame_mask, class_weight)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        batch_metrics = compute_chord_batch_metrics(chord_probs, y_chord, frame_mask, most_common_idx)
        for k in totals:
            totals[k] += batch_metrics[k]

    eps = 1e-8
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": totals["correct"] / (totals["valid"] + eps),
        "baseline": totals["baseline_correct"] / (totals["valid"] + eps),
    }


@torch.no_grad()
def evaluate_chord(model, loader, device, class_weight, most_common_idx) -> dict:
    model.eval()
    total_loss = 0.0
    totals = {"correct": 0, "baseline_correct": 0, "valid": 0}

    for x, y_chord, frame_mask in loader:
        x, y_chord, frame_mask = x.to(device), y_chord.to(device), frame_mask.to(device)

        chord_probs = model(x)
        loss = calculate_chord_loss(chord_probs, y_chord, frame_mask, class_weight)

        total_loss += loss.item() * x.size(0)
        batch_metrics = compute_chord_batch_metrics(chord_probs, y_chord, frame_mask, most_common_idx)
        for k in totals:
            totals[k] += batch_metrics[k]

    eps = 1e-8
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": totals["correct"] / (totals["valid"] + eps),
        "baseline": totals["baseline_correct"] / (totals["valid"] + eps),
    }


def train_model_chord(model, train_loader, val_loader, epochs, lr, device,
                       class_weight, most_common_idx, patience=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_chord(model, train_loader, optimizer, device, class_weight, most_common_idx)
        val_metrics = evaluate_chord(model, val_loader, device, class_weight, most_common_idx)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_chord_acc={val_metrics['accuracy']:.3f} (most_common_baseline={val_metrics['baseline']:.3f}) | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return model, history


def cross_validate_by_player_chord(
        model_fn,
        cqt_files,
        jams_files,
        time_frames=172,
        stride=129,
        batch_size=16,
        epochs=100,
        lr=0.001,
        patience=10,
        device=None,
        checkpoint_dir="output",
        checkpoint_prefix="conv2d_chord_only_checkpoint",
        extra_checkpoint_fields=None,
    ) -> list:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    file_pairs = list(zip(cqt_files, jams_files))
    key_fn = lambda pair: player_id_from_stem(os.path.basename(pair[0]))
    players = sorted({key_fn(pair) for pair in file_pairs})
    print(f"Cross-validating across {len(players)} players (task=chord): {players}")

    fold_metrics = []
    for held_out in players:
        print(f"\n{'=' * 60}\nFold: held-out player {held_out} (task=chord)\n{'=' * 60}")

        train_pairs, val_pairs = grouped_split(file_pairs, key_fn=key_fn, val_players=[held_out])
        print_and_verify_split(train_pairs, val_pairs, key_fn=key_fn, jams_fn=lambda pair: pair[1])

        train_cqt, train_jams = zip(*train_pairs)
        val_cqt, val_jams = zip(*val_pairs)

        train_ds = ChordGuitarDataset(list(train_cqt), list(train_jams), time_frames=time_frames, stride=stride)
        val_ds = ChordGuitarDataset(list(val_cqt), list(val_jams), time_frames=time_frames, stride=stride)

        print("Scanning training labels to compute chord class weights...")
        counts = compute_chord_class_counts(train_ds)
        class_weight = compute_chord_class_weights(counts).to(device)
        most_common_idx = int(counts.argmax())
        print(f"Most common chord this fold: {train_ds.vocab[most_common_idx]} ({counts[most_common_idx]/counts.sum():.1%} of frames)")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = model_fn().to(device)
        model, history = train_model_chord(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            class_weight=class_weight,
            most_common_idx=most_common_idx,
            patience=patience,
        )

        best_idx = min(range(len(history["val"])), key=lambda i: history["val"][i]["loss"])
        best_metrics = history["val"][best_idx]
        fold_metrics.append({"held_out_player": held_out, **best_metrics})

        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_holdout{held_out}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"model_state_dict": model.state_dict(), "held_out_player": held_out, "vocab": train_ds.vocab}
        if extra_checkpoint_fields:
            save_dict.update(extra_checkpoint_fields)
        torch.save(save_dict, checkpoint_path)
        print(f"Saved fold checkpoint to: {checkpoint_path}")

    print(f"\n{'=' * 60}\nCross-validation summary ({len(players)} folds, task=chord)\n{'=' * 60}")
    for m in fold_metrics:
        print(f"  player {m['held_out_player']}: val_loss={m['loss']:.4f} chord_acc={m['accuracy']:.3f} (baseline={m['baseline']:.3f})")

    for key in ["loss", "accuracy", "baseline"]:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"  {key}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return fold_metrics

# ── 4. Main Execution ────────────────────────────────────────────────

def main() -> None:
    input_bins = 252
    sr = 22050
    hop_length = 512
    time_frames = round(4 * sr / hop_length)   # 4s sliding window
    stride = round(3 * sr / hop_length)        # 3s stride -> 1s overlap
    batch_size = 16
    temporal_hidden = 256
    latent_dim = 128
    dropout = 0.4

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    cqt_files, jams_files = find_cqt_jams_pairs()

    with open(os.path.join(CHORD_CACHE_DIR, "_vocab.json")) as f:
        num_classes = len(json.load(f))

    def build_model():
        return Conv2dChordOnlyModel(
            input_bins=input_bins,
            num_classes=num_classes,
            temporal_hidden=temporal_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    cross_validate_by_player_chord(
        model_fn=build_model,
        cqt_files=cqt_files,
        jams_files=jams_files,
        time_frames=time_frames,
        stride=stride,
        batch_size=batch_size,
        epochs=100,
        lr=0.001,
        patience=10,
        device=device,
        checkpoint_dir="output",
        checkpoint_prefix="conv2d_chord_only_checkpoint",
        extra_checkpoint_fields={
            "input_bins": input_bins,
            "num_classes": num_classes,
            "temporal_hidden": temporal_hidden,
            "latent_dim": latent_dim,
            "dropout": dropout,
        },
    )

    print("Cross-validation job successful.")

if __name__ == "__main__":
    main()
