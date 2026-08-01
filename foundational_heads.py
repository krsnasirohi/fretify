import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import glob
import os
from pathlib import Path
import jams
import librosa

# ── 1. The Model ─────────────────────────────────────────

class GuitarFoundationalModel(nn.Module):
    def __init__(self, input_bins=252, latent_dim=512, num_pitches=49, num_frets=23, temporal_hidden=256):
        super().__init__()
        # 1. Shared Backbone Encoder (per-frame feature extractor)
        self.backbone = nn.Sequential(
            nn.Linear(input_bins, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
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

        # Head A: Pitch
        pitch_logits = self.pitch_head(context)
        pitch_probs = torch.sigmoid(pitch_logits)

        # Head B: Tablature
        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)

        return pitch_probs, tab_probs

# ── 2. The Dataset ───────────────────────────────────────────────────

class PrecomputedGuitarDataset(Dataset):
    """Loads pre-computed CQT tensors and parses JAMS files for targets."""
    def __init__(self, cqt_files, jams_files, time_frames=100, hop_length=512, sr=22050):
        self.cqt_files = cqt_files
        self.jams_files = jams_files
        self.time_frames = time_frames
        self.hop_length = hop_length
        self.sr = sr

        self.min_midi = 40   # Low E2
        self.max_midi = 88   # High E6 (49 classes)
        self.num_pitches = self.max_midi - self.min_midi + 1
        self.mute_class = 22
        self.num_frets = self.mute_class + 1
        # GuitarSet's note_midi annotations are ordered low string -> high
        # string (E2, A2, D3, G3, B3, E4) -- this must match that order or
        # every fret = midi - open_string comes out wrong.
        self.open_strings = [40, 45, 50, 55, 59, 64]

    def __len__(self):
        return len(self.cqt_files)

    def _build_labels(self, jams_path, total_frames):
        """Parses a JAMS file into frame-aligned pitch/tab labels."""
        Y_pitch = np.zeros((total_frames, self.num_pitches), dtype=np.float32)
        Y_tab = np.full((total_frames, 6), self.mute_class, dtype=np.int64)

        jam = jams.load(jams_path)
        note_annotations = jam.annotations.search(namespace='note_midi')

        for string_idx, string_ann in enumerate(note_annotations):
            for note in string_ann:
                start_frame = max(0, librosa.time_to_frames(note.time, sr=self.sr, hop_length=self.hop_length))
                end_frame = min(total_frames, librosa.time_to_frames(note.time + note.duration, sr=self.sr, hop_length=self.hop_length))

                midi_pitch = int(round(note.value))

                if self.min_midi <= midi_pitch <= self.max_midi:
                    pitch_idx = midi_pitch - self.min_midi
                    Y_pitch[start_frame:end_frame, pitch_idx] = 1.0

                    fret = midi_pitch - self.open_strings[string_idx]
                    if 0 <= fret <= 21:
                        Y_tab[start_frame:end_frame, string_idx] = fret

        return Y_pitch, Y_tab

    def __getitem__(self, idx):
        # 1. Load Pre-computed CQT
        cqt = np.load(self.cqt_files[idx]) # Shape: (Total_Frames, 252)
        total_frames = cqt.shape[0]

        # 2. Parse JAMS into frame-aligned labels
        Y_pitch, Y_tab = self._build_labels(self.jams_files[idx], total_frames)

        # 3. Track which frames are real vs. padding, so padded frames
        # don't get counted as "free" correct mute/silence predictions.
        valid_frames = min(total_frames, self.time_frames)
        frame_mask = np.zeros(self.time_frames, dtype=np.float32)
        frame_mask[:valid_frames] = 1.0

        # 4. Chunking/Padding (Ensures batch uniformity)
        if total_frames > self.time_frames:
            cqt = cqt[:self.time_frames, :]
            Y_pitch = Y_pitch[:self.time_frames, :]
            Y_tab = Y_tab[:self.time_frames, :]
        else:
            pad_len = self.time_frames - total_frames
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

# ── 5. Main Execution ────────────────────────────────────────────────

def main() -> None:
    input_bins = 252
    time_frames = 100
    num_pitches = 49
    num_frets = 23
    batch_size = 16

    # Target Apple Silicon GPU architecture
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Actual Data
    cqt_dir = "./output/processed_cqt"
    jams_dir = "./data/guitarset/annotation"

    raw_cqt_files = glob.glob(os.path.join(cqt_dir, "*.npy"))
    raw_jams_files = glob.glob(os.path.join(jams_dir, "*.jams"))

    if not raw_cqt_files:
        raise ValueError("No .npy files found! Run cqt.py first.")

    jams_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in raw_jams_files}

    cqt_files = []
    jams_files = []

    for cqt_path in sorted(raw_cqt_files):
        raw_name = os.path.splitext(os.path.basename(cqt_path))[0]
        clean_name = raw_name.replace("_hex", "").replace("_mic", "")

        if clean_name in jams_dict:
            cqt_files.append(cqt_path)
            jams_files.append(jams_dict[clean_name])

    print(f"Successfully paired {len(cqt_files)} CQT/JAMS combinations.")

    if len(cqt_files) == 0:
        raise ValueError("Matched 0 files! The .npy and .jams filenames still do not align.")

    # 80/20 Split
    split_idx = int(len(cqt_files) * 0.8)

    train_ds = PrecomputedGuitarDataset(cqt_files[:split_idx], jams_files[:split_idx], time_frames=time_frames)
    val_ds = PrecomputedGuitarDataset(cqt_files[split_idx:], jams_files[split_idx:], time_frames=time_frames)

    print("Scanning training labels to compute class weights (counteracts the dominant mute/silent classes)...")
    pitch_pos_weight, tab_class_weight = compute_class_weights(train_ds)
    pitch_pos_weight = pitch_pos_weight.to(device)
    tab_class_weight = tab_class_weight.to(device)
    print(f"Pitch pos_weight range: [{pitch_pos_weight.min():.2f}, {pitch_pos_weight.max():.2f}]")
    print(f"Tab class weights (0-21=fret, 22=mute): {[round(w, 2) for w in tab_class_weight.tolist()]}")

    # UPDATED: Parallel workers to dismantle the JAMS I/O disk bottleneck on your M5 Pro
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Loaded training and validation DataLoaders.")

    # 2. Build model
    temporal_hidden = 256
    model = GuitarFoundationalModel(
        input_bins=input_bins,
        latent_dim=512,
        num_pitches=num_pitches,
        num_frets=num_frets,
        temporal_hidden=temporal_hidden,
    ).to(device)

    print("Built model layout.")

    model, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.001,
        device=device,
        pitch_pos_weight=pitch_pos_weight,
        tab_class_weight=tab_class_weight,
        mute_class=train_ds.mute_class,
        patience=5
    )

    checkpoint_path = Path("output/model_checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_bins": input_bins,
        "latent_dim": 512,
        "num_pitches": num_pitches,
        "num_frets": num_frets,
        "temporal_hidden": temporal_hidden,
    }, checkpoint_path)
    print(f"Saved model checkpoint to: {checkpoint_path}")

    print("Training job successful.")

if __name__ == "__main__":
    main()
