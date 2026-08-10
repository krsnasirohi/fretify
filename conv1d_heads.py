"""Comparison architecture: replaces the MLP backbone with a small 1D CNN
that convolves across the CQT's frequency axis, applied independently per
frame (no cross-frame awareness here -- that's still the GRU's job, same
as in foundational_heads.py/gru_heads.py).

Why a CNN over frequency instead of a flat MLP: CQT bins are log-spaced (3
bins/semitone), so a note's harmonic pattern (fundamental + overtones)
looks like roughly the same shape shifted to a different position,
regardless of which pitch is playing -- that's the whole reason CQT (over
a linear-frequency STFT) gets used for pitch tasks in the first place. A
convolutional filter sliding across frequency can learn one shared
harmonic-pattern detector and reuse it at every pitch; a Linear layer has
to learn an independent weight combination per pitch instead. This is also
what both Basic Pitch and TabCNN (this project's own reference points --
see basic_pitch_heads.py and the tab-cnn repo pulled into
precompute_annotations.py's design discussion) actually do, rather than a
flat MLP.

Bonus relevant to this project's real constraint (GuitarSet is small):
conv weight sharing means fewer independent parameters for a given
receptive field than a Linear layer covering the same span, so this comes
in smaller than even SmallMLPGuitarModel despite the extra depth.

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

class Conv1dGuitarModel(nn.Module):
    """Per-frame 1D CNN over the 252 CQT frequency bins, replacing the MLP
    backbone. Frequency is progressively downsampled via pooling while
    channel depth increases, then average-pooled to a small fixed length
    (not a single scalar -- global pooling would destroy the very
    positional information pitch/tab prediction needs) and projected to
    latent_dim before the same GRU + heads used elsewhere.
    """
    def __init__(self, input_bins=252, num_pitches=49, num_frets=23, temporal_hidden=256,
                 latent_dim=128, dropout=0.3):
        super().__init__()
        self.input_bins = input_bins

        self.freq_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),                              # 252 -> 126
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),                              # 126 -> 63
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AvgPool1d(kernel_size=3, stride=3),         # 63 -> 21, keeps finer frequency position
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
        if x.dim() == 4:
            x = x.squeeze(1)

        batch_size, time_frames, _ = x.shape

        # Conv1d wants (N, C, L) -- fold time into the batch dim so every
        # frame is convolved over frequency independently.
        frames = x.reshape(batch_size * time_frames, 1, self.input_bins)
        frames = self.freq_conv(frames)
        frames = frames.flatten(1)
        latent = self.freq_proj(frames)
        latent = latent.view(batch_size, time_frames, -1)

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
    stride = round(3 * sr / hop_length)        # 3s stride -> 1s overlap
    num_pitches = 49
    num_frets = 23
    batch_size = 16
    temporal_hidden = 256
    latent_dim = 128
    dropout = 0.3

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    cqt_files, jams_files = find_cqt_jams_pairs()

    def build_model():
        return Conv1dGuitarModel(
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
        patience=5,
        device=device,
        checkpoint_dir="output",
        checkpoint_prefix="conv1d_heads_checkpoint",
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
