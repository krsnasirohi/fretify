import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import glob
import os
from pathlib import Path

from preprocessing.data_splits import grouped_split, player_id_from_stem, print_and_verify_split

# ── 1. The Model ─────────────────────────────────────────

class GuitarFoundationalModel(nn.Module):
    def __init__(self, input_bins=252, latent_dim=512, num_pitches=49, num_frets=23, temporal_hidden=256, dropout=0.3):
        super().__init__()
        # 1. Shared Backbone Encoder (per-frame feature extractor)
        self.backbone = nn.Sequential(
            nn.Linear(input_bins, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 1b. Temporal context: a small BiGRU lets each frame's prediction
        # depend on neighboring frames. This mainly helps the tab head --
        # the same pitch can usually be fretted on 2-4 different strings, and
        # disambiguating which one was actually played leans on context
        # (note transitions, playing position) that a frame-independent
        # model can't see.
        self.temporal = nn.GRU(
            input_size=latent_dim,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        context_dim = temporal_hidden * 2

        # Dropout on the GRU's output, shared by both heads. nn.GRU's own
        # `dropout` argument only applies between stacked recurrent layers
        # and is silently a no-op here since num_layers=1, so regularizing
        # the temporal output has to be this separate explicit layer. This
        # ~1.5M-param stack overfits GuitarSet's limited data without it.
        self.context_dropout = nn.Dropout(dropout)

        # 2. Head A: Acoustic Pitch Head
        self.pitch_head = nn.Linear(context_dim, num_pitches)

        # 3. Head B: Tablature Fretboard Head
        self.tab_head = nn.Linear(context_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        # x shape: (batch, time_frames, input_bins)
        if x.dim() == 4:
            x = x.squeeze(1)

        # 2. THE CRITICAL FIX: We must unpack all 3 values.
        # The '_' catches the input_bins value.
        batch_size, time_frames, _ = x.shape

        latent = self.backbone(x)
        context, _ = self.temporal(latent)
        context = self.context_dropout(context)

        # Head A: Pitch
        pitch_logits = self.pitch_head(context)
        pitch_probs = torch.sigmoid(pitch_logits)

        # Head B: Tablature
        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)

        return pitch_probs, tab_probs


class SmallMLPGuitarModel(nn.Module):
    """Middle ground between GuitarFoundationalModel's 252->256->512 MLP
    (~1.5M params, overfits GuitarSet's limited data) and gru_heads.py's
    GRUHeadsModel, which drops the MLP entirely and feeds raw 252-dim CQT
    bins straight into the GRU (~879K params, underfits -- likely because
    asking the GRU to do per-frame feature extraction AND temporal
    modeling at once is a harder optimization problem, not just a capacity
    shortfall).

    Shrinking the backbone to 252->64->64 keeps a small nonlinear per-frame
    projection (so the GRU isn't working from raw bins) while working out
    smaller overall than GRUHeadsModel: a GRU's own parameter count scales
    with its input size, so compressing 252 dims down to 64 before the GRU
    shrinks the GRU itself by more than the small backbone adds back.
    """
    def __init__(self, input_bins=252, hidden_dim=64, latent_dim=64, num_pitches=49, num_frets=23, temporal_hidden=256, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_bins, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
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

        self.pitch_head = nn.Linear(context_dim, num_pitches)
        self.tab_head = nn.Linear(context_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)

        batch_size, time_frames, _ = x.shape

        latent = self.backbone(x)
        context, _ = self.temporal(latent)
        context = self.context_dropout(context)

        pitch_logits = self.pitch_head(context)
        pitch_probs = torch.sigmoid(pitch_logits)

        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)

        return pitch_probs, tab_probs

# ── 2. The Dataset ───────────────────────────────────────────────────

class PrecomputedGuitarDataset(Dataset):
    """Loads pre-computed CQT tensors and pre-computed per-frame pitch/tab
    labels (see precompute_annotations.py) for targets.

    Each clip is tiled into overlapping `time_frames`-long sliding windows,
    `stride` frames apart, rather than truncated to just its first window
    -- clips average ~30s vs. a single short window, so keeping only the
    first window was discarding most of every clip. The final window per
    clip is anchored to end exactly at the clip's last frame (rather than
    landing at whatever the next stride multiple happens to be), so it's
    still a full, real (non-padded) window instead of a mostly-empty
    stride-aligned one -- e.g. a 30s clip with time_frames=4s/stride=3s
    yields windows at 0-4s, 3-7s, 6-10s, ..., ending anchored to the
    clip's actual end, with no window ever needing padding unless the
    whole clip is shorter than time_frames.
    """
    def __init__(self, cqt_files, jams_files, time_frames=172, stride=129, hop_length=512, sr=22050,
                 annotation_cache_dir="output/per_frame_annotations"):
        self.cqt_files = cqt_files
        self.jams_files = jams_files
        self.time_frames = time_frames
        self.stride = stride
        self.hop_length = hop_length
        self.sr = sr
        self.annotation_cache_dir = annotation_cache_dir

        self.min_midi = 40   # Low E2
        self.max_midi = 88   # High E6 (49 classes)
        self.num_pitches = self.max_midi - self.min_midi + 1
        self.mute_class = 22
        self.num_frets = self.mute_class + 1
        # GuitarSet's note_midi annotations are ordered low string -> high
        # string (E2, A2, D3, G3, B3, E4) -- this must match that order or
        # every fret = midi - open_string comes out wrong.
        self.open_strings = [40, 45, 50, 55, 59, 64]

        # (file_idx, start_frame) for every window across every clip.
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

    def _build_labels(self, jams_path, total_frames):
        """Loads this clip's precomputed per-frame pitch/tab labels from
        output/per_frame_annotations/ (built by precompute_annotations.py)
        instead of re-parsing the JAMS file live. The cache stores the full
        88-key (MIDI 21-108) pitch range, so this slices out just this
        project's 49-class guitar range (MIDI 40-88)."""
        stem = os.path.splitext(os.path.basename(jams_path))[0]
        cache_path = os.path.join(self.annotation_cache_dir, f"{stem}.npz")
        cached = np.load(cache_path)

        base_midi = int(cached["base_midi"])
        lo = self.min_midi - base_midi
        hi = self.max_midi - base_midi + 1
        Y_pitch = cached["Y_pitch88"][:total_frames, lo:hi]
        Y_tab = cached["Y_tab"][:total_frames]

        return Y_pitch, Y_tab

    def __getitem__(self, idx):
        # 1. Load this window's slice of the pre-computed CQT (mmap so we
        # only pull the pages this window actually needs, not the full clip).
        file_idx, start = self.windows[idx]
        cqt_full = np.load(self.cqt_files[file_idx], mmap_mode='r')
        total_frames = cqt_full.shape[0]
        end = min(start + self.time_frames, total_frames)
        cqt = np.array(cqt_full[start:end, :])

        # 2. Parse JAMS into frame-aligned labels for the full clip, then
        # slice out this window's span.
        Y_pitch_full, Y_tab_full = self._build_labels(self.jams_files[file_idx], total_frames)
        Y_pitch = Y_pitch_full[start:end, :]
        Y_tab = Y_tab_full[start:end, :]

        # 3. Track which frames are real vs. padding, so padded frames
        # don't get counted as "free" correct mute/silence predictions.
        valid_frames = end - start
        frame_mask = np.zeros(self.time_frames, dtype=np.float32)
        frame_mask[:valid_frames] = 1.0

        # 4. Pad the trailing window of a clip up to time_frames (Ensures batch uniformity)
        if valid_frames < self.time_frames:
            pad_len = self.time_frames - valid_frames
            cqt = np.pad(cqt, ((0, pad_len), (0, 0)))
            Y_pitch = np.pad(Y_pitch, ((0, pad_len), (0, 0)))
            Y_tab = np.pad(Y_tab, ((0, pad_len), (0, 0)), constant_values=self.mute_class)

        # Add the channel dimension for the CNN backbone: (1, time_frames, bins)
        cqt = np.expand_dims(cqt, axis=0)

        return torch.tensor(cqt, dtype=torch.float32), \
               torch.tensor(Y_pitch, dtype=torch.float32), \
               torch.tensor(Y_tab, dtype=torch.long), \
               torch.tensor(frame_mask, dtype=torch.float32)


def compute_class_weights(dataset: PrecomputedGuitarDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scans a dataset's labels once to build weights that counteract
    GuitarSet's dominant mute/silent classes (~71% of string-frames across
    the dataset are muted, and most of the 49 pitch classes are silent at
    any given frame). Without this, the loss is easy to minimize by mostly
    predicting silence.
    """
    pitch_pos = np.zeros(dataset.num_pitches, dtype=np.int64)
    pitch_total_frames = 0
    tab_counts = np.zeros(dataset.num_frets, dtype=np.int64)

    for cqt_path, jams_path in zip(dataset.cqt_files, dataset.jams_files):
        total_frames = np.load(cqt_path, mmap_mode='r').shape[0]
        Y_pitch, Y_tab = dataset._build_labels(jams_path, total_frames)

        pitch_pos += Y_pitch.sum(axis=0).astype(np.int64)
        pitch_total_frames += total_frames
        tab_counts += np.bincount(Y_tab.ravel(), minlength=dataset.num_frets)

    pitch_neg = pitch_total_frames - pitch_pos
    pitch_pos_weight = np.clip(pitch_neg / np.maximum(pitch_pos, 1), 1.0, 50.0)

    tab_freq = tab_counts / tab_counts.sum()
    tab_class_weight = np.clip(1.0 / np.maximum(tab_freq, 1e-4), 0.0, 20.0)
    tab_class_weight = tab_class_weight / tab_class_weight.mean()

    return (
        torch.tensor(pitch_pos_weight, dtype=torch.float32),
        torch.tensor(tab_class_weight, dtype=torch.float32),
    )

# ── 3. Train Model from Stress_AI ───────────────────────────────────

def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        device: torch.device,
        pitch_pos_weight: torch.Tensor,
        tab_class_weight: torch.Tensor,
        mute_class: int,
        patience: int = 10,
    ) -> Tuple[nn.Module, dict]:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, pitch_pos_weight, tab_class_weight, mute_class
        )
        val_metrics = evaluate(
            model, val_loader, device, pitch_pos_weight, tab_class_weight, mute_class
        )
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
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} (pitch={train_metrics['loss_pitch']:.4f} tab={train_metrics['loss_tab']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} (pitch={val_metrics['loss_pitch']:.4f} tab={val_metrics['loss_tab']:.4f}) | "
            f"val_tab_acc={val_metrics['tab_accuracy']:.3f} (mute_baseline={val_metrics['tab_baseline']:.3f}) | "
            f"val_pitch_f1={val_metrics['pitch_f1']:.3f} (P={val_metrics['pitch_precision']:.3f} R={val_metrics['pitch_recall']:.3f}) | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return model, history

# ── 4. Training & Evaluation Logic ───────────────────────────────────

def calculate_mtl_loss(pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, pitch_pos_weight, tab_class_weight):
    """Combined masked, class-weighted loss for both heads.

    frame_mask zeroes out padded frames so they don't dilute the loss, and
    the weights counteract GuitarSet's dominant mute/silent classes.
    """
    eps = 1e-7
    mask = frame_mask.unsqueeze(-1)  # (batch, time, 1)

    # Pitch: weighted BCE -- pos_weight boosts the rare "note is on" cases
    p = pitch_probs.clamp(eps, 1 - eps)
    pw = pitch_pos_weight.view(1, 1, -1)
    pitch_elem = -(pw * y_pitch * torch.log(p) + (1 - y_pitch) * torch.log(1 - p))
    loss_pitch = (pitch_elem * mask).sum() / (mask.sum() * y_pitch.shape[-1] + eps)

    # Tab: class-weighted NLL -- down-weights the dominant "mute" class
    tab_probs_permuted = tab_probs.permute(0, 3, 1, 2).contiguous()
    log_probs = torch.log(tab_probs_permuted.clamp(min=eps))
    nll_elem = F.nll_loss(log_probs, y_tab, weight=tab_class_weight, reduction='none')
    loss_tab = (nll_elem * mask).sum() / (mask.sum() * y_tab.shape[-1] + eps)

    return loss_pitch + loss_tab, loss_pitch, loss_tab


@torch.no_grad()
def compute_batch_metrics(pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, mute_class):
    """Masked metrics for one batch, including the trivial 'always mute'
    baseline so tab accuracy can be judged against it rather than in a
    vacuum."""
    mask = frame_mask.bool()
    mask_tab = mask.unsqueeze(-1).expand_as(y_tab)
    mask_pitch = mask.unsqueeze(-1).expand_as(y_pitch)

    tab_pred = tab_probs.argmax(dim=-1)
    tab_correct = ((tab_pred == y_tab) & mask_tab).sum().item()
    tab_baseline_correct = ((y_tab == mute_class) & mask_tab).sum().item()
    tab_valid = mask_tab.sum().item()

    pitch_pred = pitch_probs > 0.5
    y_pitch_bool = y_pitch.bool()
    pitch_tp = (pitch_pred & y_pitch_bool & mask_pitch).sum().item()
    pitch_fp = (pitch_pred & ~y_pitch_bool & mask_pitch).sum().item()
    pitch_fn = (~pitch_pred & y_pitch_bool & mask_pitch).sum().item()

    return {
        "tab_correct": tab_correct,
        "tab_baseline_correct": tab_baseline_correct,
        "tab_valid": tab_valid,
        "pitch_tp": pitch_tp,
        "pitch_fp": pitch_fp,
        "pitch_fn": pitch_fn,
    }


def _new_metric_totals() -> dict:
    return {
        "tab_correct": 0, "tab_baseline_correct": 0, "tab_valid": 0,
        "pitch_tp": 0, "pitch_fp": 0, "pitch_fn": 0,
    }


def _summarize_metrics(totals: dict, loss_sum: float, loss_pitch_sum: float, loss_tab_sum: float, n_examples: int) -> dict:
    eps = 1e-8
    precision = totals["pitch_tp"] / (totals["pitch_tp"] + totals["pitch_fp"] + eps)
    recall = totals["pitch_tp"] / (totals["pitch_tp"] + totals["pitch_fn"] + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "loss": loss_sum / n_examples,
        "loss_pitch": loss_pitch_sum / n_examples,
        "loss_tab": loss_tab_sum / n_examples,
        "tab_accuracy": totals["tab_correct"] / (totals["tab_valid"] + eps),
        "tab_baseline": totals["tab_baseline_correct"] / (totals["tab_valid"] + eps),
        "pitch_precision": precision,
        "pitch_recall": recall,
        "pitch_f1": f1,
    }


def train_one_epoch(model, loader, optimizer, device, pitch_pos_weight, tab_class_weight, mute_class) -> dict:
    model.train()
    total_loss = 0.0
    total_loss_pitch = 0.0
    total_loss_tab = 0.0
    totals = _new_metric_totals()

    for x, y_pitch, y_tab, frame_mask in loader:
        x, y_pitch, y_tab, frame_mask = (
            x.to(device), y_pitch.to(device), y_tab.to(device), frame_mask.to(device)
        )

        optimizer.zero_grad()
        pitch_probs, tab_probs = model(x)
        loss, loss_pitch, loss_tab = calculate_mtl_loss(
            pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, pitch_pos_weight, tab_class_weight
        )

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_loss_pitch += loss_pitch.item() * x.size(0)
        total_loss_tab += loss_tab.item() * x.size(0)
        batch_metrics = compute_batch_metrics(pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, mute_class)
        for k in totals:
            totals[k] += batch_metrics[k]

    return _summarize_metrics(totals, total_loss, total_loss_pitch, total_loss_tab, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device, pitch_pos_weight, tab_class_weight, mute_class) -> dict:
    model.eval()
    total_loss = 0.0
    total_loss_pitch = 0.0
    total_loss_tab = 0.0
    totals = _new_metric_totals()

    for x, y_pitch, y_tab, frame_mask in loader:
        x, y_pitch, y_tab, frame_mask = (
            x.to(device), y_pitch.to(device), y_tab.to(device), frame_mask.to(device)
        )

        pitch_probs, tab_probs = model(x)
        loss, loss_pitch, loss_tab = calculate_mtl_loss(
            pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, pitch_pos_weight, tab_class_weight
        )

        total_loss += loss.item() * x.size(0)
        total_loss_pitch += loss_pitch.item() * x.size(0)
        total_loss_tab += loss_tab.item() * x.size(0)
        batch_metrics = compute_batch_metrics(pitch_probs, tab_probs, y_pitch, y_tab, frame_mask, mute_class)
        for k in totals:
            totals[k] += batch_metrics[k]

    return _summarize_metrics(totals, total_loss, total_loss_pitch, total_loss_tab, len(loader.dataset))

# ── 5. Data pairing & cross-validation ───────────────────────────────

def find_cqt_jams_pairs(cqt_dir="./output/processed_cqt", jams_dir="./data/guitarset/annotation"):
    """Pairs each debleeded hex-pickup recording (audio_hex-pickup_debleeded,
    filename suffix _hex_cln) with its .jams annotation -- one clean
    recording per performance, not all 4 mic/mix/hex/hex_cln variants."""
    raw_cqt_files = glob.glob(os.path.join(cqt_dir, "*.npy"))
    raw_jams_files = glob.glob(os.path.join(jams_dir, "*.jams"))

    if not raw_cqt_files:
        raise ValueError("No .npy files found! Run cqt.py first.")

    jams_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_jams_files}

    suffix = "_hex_cln"
    cqt_files = []
    jams_files = []
    for cqt_path in sorted(raw_cqt_files):
        raw_name = os.path.splitext(os.path.basename(cqt_path))[0]
        if not raw_name.endswith(suffix):
            continue
        clean_name = raw_name[: -len(suffix)]
        if clean_name in jams_dict:
            cqt_files.append(cqt_path)
            jams_files.append(jams_dict[clean_name])

    print(f"Successfully paired {len(cqt_files)} CQT/JAMS combinations (debleeded hex variant only).")
    if len(cqt_files) == 0:
        raise ValueError("Matched 0 files! The .npy and .jams filenames still do not align.")

    return cqt_files, jams_files


def cross_validate_by_player(
        model_fn,
        cqt_files,
        jams_files,
        time_frames=172,
        stride=129,
        batch_size=16,
        epochs=100,
        lr=0.001,
        patience=5,
        device=None,
        checkpoint_dir="output",
        checkpoint_prefix="model_checkpoint",
        extra_checkpoint_fields=None,
    ) -> list:
    """Trains and evaluates one model per held-out player (GuitarSet has 6),
    instead of a single train/val split. A single held-out player is a
    high-variance, one-sample estimate of "does this generalize to an
    unseen guitarist" -- rotating the held-out player across all 6 and
    averaging gives a much more trustworthy estimate of that, at the cost
    of 6x the training time. model_fn() must return a fresh, untrained
    model on each call.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    file_pairs = list(zip(cqt_files, jams_files))
    key_fn = lambda pair: player_id_from_stem(os.path.basename(pair[0]))
    players = sorted({key_fn(pair) for pair in file_pairs})
    print(f"Cross-validating across {len(players)} players: {players}")

    fold_metrics = []
    for held_out in players:
        print(f"\n{'=' * 60}\nFold: held-out player {held_out}\n{'=' * 60}")

        train_pairs, val_pairs = grouped_split(file_pairs, key_fn=key_fn, val_players=[held_out])
        print_and_verify_split(train_pairs, val_pairs, key_fn=key_fn, jams_fn=lambda pair: pair[1])

        train_cqt, train_jams = zip(*train_pairs)
        val_cqt, val_jams = zip(*val_pairs)

        train_ds = PrecomputedGuitarDataset(list(train_cqt), list(train_jams), time_frames=time_frames, stride=stride)
        val_ds = PrecomputedGuitarDataset(list(val_cqt), list(val_jams), time_frames=time_frames, stride=stride)

        print("Scanning training labels to compute class weights...")
        pitch_pos_weight, tab_class_weight = compute_class_weights(train_ds)
        pitch_pos_weight = pitch_pos_weight.to(device)
        tab_class_weight = tab_class_weight.to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = model_fn().to(device)
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            pitch_pos_weight=pitch_pos_weight,
            tab_class_weight=tab_class_weight,
            mute_class=train_ds.mute_class,
            patience=patience,
        )

        # train_model already restores the best-epoch weights into `model`;
        # pull that same epoch's logged metrics rather than re-running eval.
        best_idx = min(range(len(history["val"])), key=lambda i: history["val"][i]["loss"])
        best_metrics = history["val"][best_idx]
        fold_metrics.append({"held_out_player": held_out, **best_metrics})

        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_holdout{held_out}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"model_state_dict": model.state_dict(), "held_out_player": held_out}
        if extra_checkpoint_fields:
            save_dict.update(extra_checkpoint_fields)
        torch.save(save_dict, checkpoint_path)
        print(f"Saved fold checkpoint to: {checkpoint_path}")

    print(f"\n{'=' * 60}\nCross-validation summary ({len(players)} folds)\n{'=' * 60}")
    for m in fold_metrics:
        print(
            f"  player {m['held_out_player']}: val_loss={m['loss']:.4f} "
            f"tab_acc={m['tab_accuracy']:.3f} (baseline={m['tab_baseline']:.3f}) "
            f"pitch_f1={m['pitch_f1']:.3f}"
        )

    for key in ["loss", "tab_accuracy", "tab_baseline", "pitch_f1", "pitch_precision", "pitch_recall"]:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"  {key}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return fold_metrics

# ── 6. Single-Task Training & Evaluation (pitch-only or tab-only models) ──
#
# Everything above assumes a model returns (pitch_probs, tab_probs). A
# single-task model (see conv2d_pitch_only.py / conv2d_tab_only.py) returns
# just one probs tensor, so it needs its own loss/metrics/train loop rather
# than reusing calculate_mtl_loss/train_model/cross_validate_by_player
# directly -- those are left untouched since 5 other scripts depend on
# their exact (pitch_probs, tab_probs) contract.

def calculate_single_task_loss(probs, y, frame_mask, task, pitch_pos_weight=None, tab_class_weight=None):
    """Same loss math as calculate_mtl_loss's per-head terms, just for
    whichever single task this model has. task must be "pitch" or "tab"."""
    eps = 1e-7
    mask = frame_mask.unsqueeze(-1)

    if task == "pitch":
        p = probs.clamp(eps, 1 - eps)
        pw = pitch_pos_weight.view(1, 1, -1)
        elem = -(pw * y * torch.log(p) + (1 - y) * torch.log(1 - p))
        return (elem * mask).sum() / (mask.sum() * y.shape[-1] + eps)
    elif task == "tab":
        probs_permuted = probs.permute(0, 3, 1, 2).contiguous()
        log_probs = torch.log(probs_permuted.clamp(min=eps))
        nll_elem = F.nll_loss(log_probs, y, weight=tab_class_weight, reduction='none')
        return (nll_elem * mask).sum() / (mask.sum() * y.shape[-1] + eps)
    else:
        raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def compute_single_task_batch_metrics(probs, y, frame_mask, task, mute_class=None):
    mask = frame_mask.bool()
    mask_y = mask.unsqueeze(-1).expand_as(y)

    if task == "pitch":
        pred = probs > 0.5
        y_bool = y.bool()
        return {
            "tp": (pred & y_bool & mask_y).sum().item(),
            "fp": (pred & ~y_bool & mask_y).sum().item(),
            "fn": (~pred & y_bool & mask_y).sum().item(),
        }
    elif task == "tab":
        pred = probs.argmax(dim=-1)
        return {
            "correct": ((pred == y) & mask_y).sum().item(),
            "baseline_correct": ((y == mute_class) & mask_y).sum().item(),
            "valid": mask_y.sum().item(),
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def _new_single_task_metric_totals(task: str) -> dict:
    if task == "pitch":
        return {"tp": 0, "fp": 0, "fn": 0}
    return {"correct": 0, "baseline_correct": 0, "valid": 0}


def _summarize_single_task_metrics(totals: dict, loss_sum: float, n_examples: int, task: str) -> dict:
    eps = 1e-8
    result = {"loss": loss_sum / n_examples}
    if task == "pitch":
        precision = totals["tp"] / (totals["tp"] + totals["fp"] + eps)
        recall = totals["tp"] / (totals["tp"] + totals["fn"] + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        result.update({"precision": precision, "recall": recall, "f1": f1})
    else:
        result.update({
            "accuracy": totals["correct"] / (totals["valid"] + eps),
            "baseline": totals["baseline_correct"] / (totals["valid"] + eps),
        })
    return result


def train_one_epoch_single_task(model, loader, optimizer, device, task, pitch_pos_weight, tab_class_weight, mute_class) -> dict:
    model.train()
    total_loss = 0.0
    totals = _new_single_task_metric_totals(task)

    for x, y_pitch, y_tab, frame_mask in loader:
        y = y_pitch if task == "pitch" else y_tab
        x, y, frame_mask = x.to(device), y.to(device), frame_mask.to(device)

        optimizer.zero_grad()
        probs = model(x)
        loss = calculate_single_task_loss(probs, y, frame_mask, task, pitch_pos_weight, tab_class_weight)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        batch_metrics = compute_single_task_batch_metrics(probs, y, frame_mask, task, mute_class)
        for k in totals:
            totals[k] += batch_metrics[k]

    return _summarize_single_task_metrics(totals, total_loss, len(loader.dataset), task)


@torch.no_grad()
def evaluate_single_task(model, loader, device, task, pitch_pos_weight, tab_class_weight, mute_class) -> dict:
    model.eval()
    total_loss = 0.0
    totals = _new_single_task_metric_totals(task)

    for x, y_pitch, y_tab, frame_mask in loader:
        y = y_pitch if task == "pitch" else y_tab
        x, y, frame_mask = x.to(device), y.to(device), frame_mask.to(device)

        probs = model(x)
        loss = calculate_single_task_loss(probs, y, frame_mask, task, pitch_pos_weight, tab_class_weight)

        total_loss += loss.item() * x.size(0)
        batch_metrics = compute_single_task_batch_metrics(probs, y, frame_mask, task, mute_class)
        for k in totals:
            totals[k] += batch_metrics[k]

    return _summarize_single_task_metrics(totals, total_loss, len(loader.dataset), task)


def train_model_single_task(
        model, train_loader, val_loader, epochs, lr, device, task,
        pitch_pos_weight, tab_class_weight, mute_class, patience=10,
    ) -> Tuple[nn.Module, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch_single_task(
            model, train_loader, optimizer, device, task, pitch_pos_weight, tab_class_weight, mute_class
        )
        val_metrics = evaluate_single_task(
            model, val_loader, device, task, pitch_pos_weight, tab_class_weight, mute_class
        )
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

        if task == "pitch":
            print(
                f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
                f"val_pitch_f1={val_metrics['f1']:.3f} (P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f}) | "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
                f"val_tab_acc={val_metrics['accuracy']:.3f} (mute_baseline={val_metrics['baseline']:.3f}) | "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return model, history


def cross_validate_by_player_single_task(
        model_fn,
        cqt_files,
        jams_files,
        task,
        time_frames=172,
        stride=129,
        batch_size=16,
        epochs=100,
        lr=0.001,
        patience=5,
        device=None,
        checkpoint_dir="output",
        checkpoint_prefix="model_checkpoint",
        extra_checkpoint_fields=None,
    ) -> list:
    """Same player-held-out cross-validation as cross_validate_by_player,
    for a single-task model that returns one probs tensor instead of
    (pitch_probs, tab_probs). task must be "pitch" or "tab".
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    file_pairs = list(zip(cqt_files, jams_files))
    key_fn = lambda pair: player_id_from_stem(os.path.basename(pair[0]))
    players = sorted({key_fn(pair) for pair in file_pairs})
    print(f"Cross-validating across {len(players)} players (task={task}): {players}")

    fold_metrics = []
    for held_out in players:
        print(f"\n{'=' * 60}\nFold: held-out player {held_out} (task={task})\n{'=' * 60}")

        train_pairs, val_pairs = grouped_split(file_pairs, key_fn=key_fn, val_players=[held_out])
        print_and_verify_split(train_pairs, val_pairs, key_fn=key_fn, jams_fn=lambda pair: pair[1])

        train_cqt, train_jams = zip(*train_pairs)
        val_cqt, val_jams = zip(*val_pairs)

        train_ds = PrecomputedGuitarDataset(list(train_cqt), list(train_jams), time_frames=time_frames, stride=stride)
        val_ds = PrecomputedGuitarDataset(list(val_cqt), list(val_jams), time_frames=time_frames, stride=stride)

        print("Scanning training labels to compute class weights...")
        pitch_pos_weight, tab_class_weight = compute_class_weights(train_ds)
        pitch_pos_weight = pitch_pos_weight.to(device)
        tab_class_weight = tab_class_weight.to(device)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = model_fn().to(device)
        model, history = train_model_single_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            task=task,
            pitch_pos_weight=pitch_pos_weight,
            tab_class_weight=tab_class_weight,
            mute_class=train_ds.mute_class,
            patience=patience,
        )

        best_idx = min(range(len(history["val"])), key=lambda i: history["val"][i]["loss"])
        best_metrics = history["val"][best_idx]
        fold_metrics.append({"held_out_player": held_out, **best_metrics})

        checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_holdout{held_out}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {"model_state_dict": model.state_dict(), "held_out_player": held_out, "task": task}
        if extra_checkpoint_fields:
            save_dict.update(extra_checkpoint_fields)
        torch.save(save_dict, checkpoint_path)
        print(f"Saved fold checkpoint to: {checkpoint_path}")

    print(f"\n{'=' * 60}\nCross-validation summary ({len(players)} folds, task={task})\n{'=' * 60}")
    for m in fold_metrics:
        if task == "pitch":
            print(f"  player {m['held_out_player']}: val_loss={m['loss']:.4f} pitch_f1={m['f1']:.3f} (P={m['precision']:.3f} R={m['recall']:.3f})")
        else:
            print(f"  player {m['held_out_player']}: val_loss={m['loss']:.4f} tab_acc={m['accuracy']:.3f} (baseline={m['baseline']:.3f})")

    metric_keys = ["loss", "f1", "precision", "recall"] if task == "pitch" else ["loss", "accuracy", "baseline"]
    for key in metric_keys:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"  {key}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return fold_metrics

# ── 7. Main Execution ────────────────────────────────────────────────

def main() -> None:
    input_bins = 252
    sr = 22050
    hop_length = 512
    time_frames = round(4 * sr / hop_length)   # 4s sliding window
    stride = round(3 * sr / hop_length)        # 3s stride -> 1s overlap
    num_pitches = 49
    num_frets = 23
    batch_size = 16
    temporal_hidden = 256
    hidden_dim = 64
    latent_dim = 64
    dropout = 0.3

    # Target Apple Silicon GPU architecture
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    cqt_files, jams_files = find_cqt_jams_pairs()

    def build_model():
        return SmallMLPGuitarModel(
            input_bins=input_bins,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_pitches=num_pitches,
            num_frets=num_frets,
            temporal_hidden=temporal_hidden,
            dropout=dropout,
        )

    cross_validate_by_player(
        model_fn=build_model,
        cqt_files=cqt_files,
        jams_files=jams_files,
        time_frames=time_frames,
        stride=stride,
        batch_size=batch_size,
        epochs=100,
        lr=0.001,
        patience=5,
        device=device,
        checkpoint_dir="output",
        checkpoint_prefix="small_mlp_checkpoint",
        extra_checkpoint_fields={
            "input_bins": input_bins,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_pitches": num_pitches,
            "num_frets": num_frets,
            "temporal_hidden": temporal_hidden,
            "dropout": dropout,
        },
    )

    print("Cross-validation job successful.")

if __name__ == "__main__":
    main()
