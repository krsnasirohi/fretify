import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np
import glob
import os
import jams
import librosa

# ── 1. The Model ─────────────────────────────────────────

class GuitarFoundationalModel(nn.Module):
    def __init__(self, input_bins=252, latent_dim=512, num_pitches=49, num_frets=23):
        super().__init__()
        # 1. Shared Backbone Encoder
        self.backbone = nn.Sequential(
            nn.Linear(input_bins, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        
        # 2. Head A: Acoustic Pitch Head
        self.pitch_head = nn.Linear(latent_dim, num_pitches)
        
        # 3. Head B: Tablature Fretboard Head
        self.tab_head = nn.Linear(latent_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        # x shape: (batch, time_frames, input_bins)
        batch_size, time_frames, _ = x.shape
        
        latent = self.backbone(x)
        
        # Head A: Pitch
        pitch_logits = self.pitch_head(latent)
        pitch_probs = torch.sigmoid(pitch_logits)
        
        # Head B: Tablature
        tab_logits = self.tab_head(latent)
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
        self.mute_class = 22 
        self.open_strings = [64, 59, 55, 50, 45, 40]

    def __len__(self):
        return len(self.cqt_files)

    def __getitem__(self, idx):
        # 1. Load Pre-computed CQT
        cqt = np.load(self.cqt_files[idx]) # Shape: (Total_Frames, 252)
        total_frames = cqt.shape[0]

        # 2. Initialize Labels
        Y_pitch = np.zeros((total_frames, 49), dtype=np.float32)
        Y_tab = np.full((total_frames, 6), self.mute_class, dtype=np.int64)

        # 3. Parse JAMS
        jam = jams.load(self.jams_files[idx])
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

        # 4. Chunking/Padding (Ensures batch uniformity)
        if total_frames > self.time_frames:
            # Slices the first N frames. (Add random cropping here later for robustness)
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
               torch.tensor(Y_tab, dtype=torch.long)

# ── 3. Train  Model from Stress_AI ───────────────────────────────────

def train_model(
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        device: torch.device,
        patience: int = 10,
    ) -> Tuple[nn.Module, dict]:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train_acc={train_acc:.3f} val_acc={val_acc:.3f} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nLoaded best model from epoch {best_epoch} (val_acc={best_val_acc:.3f})")
    return model, history

# ── 4. Training & Evaluation Logic ───────────────────────────────────

def calculate_mtl_loss(pitch_probs, tab_probs, y_pitch, y_tab):
    """Calculates the combined loss for both heads."""
    # 1. Pitch Loss (Multi-label classification -> Binary Cross Entropy)
    loss_pitch = F.binary_cross_entropy(pitch_probs, y_pitch)
    
    # 2. Tab Loss (Multi-class per string -> Negative Log Likelihood)
    # PyTorch expects classes in the 2nd dimension for NLL/CrossEntropy: (batch, classes, time, strings)
    # So we permute: (batch, time, string, fret) -> (batch, fret, time, string)
    tab_probs_permuted = tab_probs.permute(0, 3, 1, 2)
    # Add a small epsilon to prevent log(0)
    loss_tab = F.nll_loss(torch.log(tab_probs_permuted + 1e-8), y_tab)
    
    # 3. Combine them (you can weight these differently if one dominates)
    return loss_pitch + loss_tab


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for x, y_pitch, y_tab in loader:
        x, y_pitch, y_tab = x.to(device), y_pitch.to(device), y_tab.to(device)
        
        optimizer.zero_grad()

        # Forward pass (only 1 input now)
        pitch_probs, tab_probs = model(x)
        
        # Calculate composite loss
        loss = calculate_mtl_loss(pitch_probs, tab_probs, y_pitch, y_tab)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0

    for x, y_pitch, y_tab in loader:
        x, y_pitch, y_tab = x.to(device), y_pitch.to(device), y_tab.to(device)
        
        pitch_probs, tab_probs = model(x)
        loss = calculate_mtl_loss(pitch_probs, tab_probs, y_pitch, y_tab)
        
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

# ── 5. Main Execution ────────────────────────────────────────────────

def main() -> None:
    input_bins = 252
    time_frames = 100
    num_pitches = 49
    num_frets = 23
    batch_size = 16
    epochs = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Actual Data
    cqt_dir = "./data/guitarset/processed_cqt"
    jams_dir = "./data/guitarset/annotation"
    
    # Ensure they are sorted so they align perfectly
    cqt_files = sorted(glob.glob(os.path.join(cqt_dir, "*.npy")))
    jams_files = sorted(glob.glob(os.path.join(jams_dir, "*.jams")))
    
    if not cqt_files:
        raise ValueError("No .npy files found! Run cqt.py first.")

    # 80/20 Split
    split_idx = int(len(cqt_files) * 0.8)
    
    train_ds = PrecomputedGuitarDataset(cqt_files[:split_idx], jams_files[:split_idx], time_frames=time_frames)
    val_ds = PrecomputedGuitarDataset(cqt_files[split_idx:], jams_files[split_idx:], time_frames=time_frames)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Build model
    model = GuitarFoundationalModel(
        input_bins=input_bins, 
        latent_dim=512, 
        num_pitches=num_pitches, 
        num_frets=num_frets
    ).to(device)

    # Note: Passed the remaining arguments exactly as they were in your train_model function
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=0.001,
        device=device
    )

if __name__ == "__main__":
    main()