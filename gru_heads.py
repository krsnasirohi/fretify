"""Simplified comparison architecture: drops GuitarFoundationalModel's MLP
backbone and feeds raw CQT frames straight into the GRU, since a GRU already
applies its own linear transform to its input at each timestep -- the MLP's
job (per-frame nonlinear feature extraction before temporal mixing) isn't
strictly required, just usually helpful when there's enough data to support
the extra capacity.

Given GuitarSet is small (~600 train examples after the player-grouped
split, vs. GuitarFoundationalModel's ~1.5M params), this lower-capacity
model is worth comparing against it directly.

Reuses PrecomputedGuitarDataset, compute_class_weights, and train_model from
foundational_heads.py -- only the model architecture changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundational_heads import find_cqt_jams_pairs, cross_validate_by_player

# ── 1. The Model ─────────────────────────────────────────

class GRUHeadsModel(nn.Module):
    """No MLP backbone -- the GRU consumes raw CQT bins directly, followed
    by the same pitch/tab head shapes as GuitarFoundationalModel."""
    def __init__(self, input_bins=252, num_pitches=49, num_frets=23, temporal_hidden=256):
        super().__init__()
        self.temporal = nn.GRU(
            input_size=input_bins,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        context_dim = temporal_hidden * 2

        self.pitch_head = nn.Linear(context_dim, num_pitches)
        self.tab_head = nn.Linear(context_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        # x shape: (batch, time_frames, input_bins)
        if x.dim() == 4:
            x = x.squeeze(1)

        batch_size, time_frames, _ = x.shape

        context, _ = self.temporal(x)

        pitch_logits = self.pitch_head(context)
        pitch_probs = torch.sigmoid(pitch_logits)

        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)

        return pitch_probs, tab_probs

# ── 2. Main Execution ────────────────────────────────────────────────

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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    cqt_files, jams_files = find_cqt_jams_pairs()

    def build_model():
        return GRUHeadsModel(
            input_bins=input_bins,
            num_pitches=num_pitches,
            num_frets=num_frets,
            temporal_hidden=temporal_hidden,
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
        checkpoint_prefix="gru_heads_checkpoint",
        extra_checkpoint_fields={
            "input_bins": input_bins,
            "num_pitches": num_pitches,
            "num_frets": num_frets,
            "temporal_hidden": temporal_hidden,
        },
    )

    print("Cross-validation job successful.")

if __name__ == "__main__":
    main()
