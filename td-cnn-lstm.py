import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import numpy as np

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

class GuitarDataset(Dataset):
    """A custom dataset to hold our guitar audio bins, pitch labels, and tab labels."""
    def __init__(self, x_data, y_pitch, y_tab):
        self.x_data = torch.FloatTensor(x_data)
        self.y_pitch = torch.FloatTensor(y_pitch) # Float for BCE Loss
        self.y_tab = torch.LongTensor(y_tab)      # Long for NLL Loss (class indices)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_pitch[idx], self.y_tab[idx]

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
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
    # Hyperparameters mapping to the model defaults
    input_bins = 252
    time_frames = 100
    num_pitches = 49
    num_frets = 23
    batch_size = 16
    epochs = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate Dummy Data (Replace this with your actual data loading logic)
    print("Generating dummy data...")
    num_samples = 100
    
    # X shape: (samples, time_frames, input_bins)
    dummy_x = np.random.randn(num_samples, time_frames, input_bins)
    
    # Y Pitch shape: (samples, time_frames, num_pitches) -> Multi-hot encoded
    dummy_y_pitch = np.random.randint(0, 2, size=(num_samples, time_frames, num_pitches))
    
    # Y Tab shape: (samples, time_frames, 6 strings) -> Values are fret indices (0 to 22)
    dummy_y_tab = np.random.randint(0, num_frets, size=(num_samples, time_frames, 6))

    # Split into train/val
    train_ds = GuitarDataset(dummy_x[:80], dummy_y_pitch[:80], dummy_y_tab[:80])
    val_ds = GuitarDataset(dummy_x[80:], dummy_y_pitch[80:], dummy_y_tab[80:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Build model with correct arguments
    model = GuitarFoundationalModel(
        input_bins=input_bins, 
        latent_dim=512, 
        num_pitches=num_pitches, 
        num_frets=num_frets
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 3. Training Loop
    print("\nStarting Training...")
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save logic here if desired
            
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()