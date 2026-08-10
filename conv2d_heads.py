"""Comparison architecture: replaces the MLP/1D-conv backbone with a 2D CNN
that convolves jointly over time AND frequency (closer to TabCNN's actual
design -- see the tab-cnn repo referenced from precompute_annotations.py's
design discussion), instead of processing each frame independently.

Unlike conv1d_heads.py (zero cross-frame awareness until the GRU) or the
MLP backbones (same), this backbone's own kernels span a handful of
neighboring frames directly, so it can pick up local time-frequency
patterns (onsets, transients) before the GRU ever sees the data. Pooling
is applied only along the frequency axis (via kernel_size=(1, k)), never
time, so every one of the window's frames still gets its own feature
vector afterward -- this task needs frame-level pitch/tab predictions
across the whole window, not one classification per window the way
TabCNN's simpler fixed-window design does.

Reuses PrecomputedGuitarDataset (with its sliding-window tiling),
compute_class_weights, train_model, find_cqt_jams_pairs, and
cross_validate_by_player from foundational_heads.py -- only the backbone
architecture changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundational_heads import find_cqt_jams_pairs, cross_validate_by_player

# ── 1. The Model ─────────────────────────────────────────

class Conv2dGuitarModel(nn.Module):
    """Joint time-frequency 2D CNN backbone. Time padding keeps every
    frame's own output aligned 1:1 with its input frame ('same' in time);
    frequency is progressively downsampled via (1, k)-shaped pooling,
    which never touches the time axis, then average-pooled to a small
    fixed length and projected to latent_dim before the same GRU + heads
    used elsewhere.
    """
    def __init__(self, input_bins=252, num_pitches=49, num_frets=23, temporal_hidden=256,
                 latent_dim=128, dropout=0.3):
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
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),  # freq 63 -> 21, time untouched
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

        self.pitch_head = nn.Linear(context_dim, num_pitches)
        self.tab_head = nn.Linear(context_dim, 6 * num_frets)
        self.num_frets = num_frets
        self.num_pitches = num_pitches

    def forward(self, x):
        # x shape: (batch, 1, time_frames, input_bins) or (batch, time_frames, input_bins)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, _, time_frames, _ = x.shape

        feat = self.conv(x)                       # (batch, 32, time_frames, 8)
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size, time_frames, -1)
        latent = self.freq_proj(feat)

        context, _ = self.temporal(latent)
        context = self.context_dropout(context)

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
    stride = round(1 * sr / hop_length)        # 1s stride -> 0s overlap
    num_pitches = 49
    num_frets = 23
    batch_size = 16
    temporal_hidden = 256
    latent_dim = 128
    dropout = 0.4

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    cqt_files, jams_files = find_cqt_jams_pairs()

    def build_model():
        return Conv2dGuitarModel(
            input_bins=input_bins,
            num_pitches=num_pitches,
            num_frets=num_frets,
            temporal_hidden=temporal_hidden,
            latent_dim=latent_dim,
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
        patience=10,
        device=device,
        checkpoint_dir="output",
        checkpoint_prefix="conv2d_heads_checkpoint",
        extra_checkpoint_fields={
            "input_bins": input_bins,
            "num_pitches": num_pitches,
            "num_frets": num_frets,
            "temporal_hidden": temporal_hidden,
            "latent_dim": latent_dim,
            "dropout": dropout,
        },
    )

    print("Cross-validation job successful.")

if __name__ == "__main__":
    main()
