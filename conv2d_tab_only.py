"""Single-task comparison architecture: the same joint time-frequency 2D
CNN backbone as conv2d_heads.py, but with only a tab_head -- no
pitch_head, no pitch loss, no pitch labels used at all. Predicts fret (or
mute) for each of the 6 strings, nothing about which MIDI pitch.

This is a genuinely single-task model, not conv2d_heads.py's dual-head
model with one head ignored -- doing it that way would still compute a
meaningless pitch loss term that pollutes total val_loss (and therefore
which epoch cross-validation picks as "best"). Instead this uses the
single-task loss/metrics/training loop added to foundational_heads.py
(calculate_single_task_loss, train_model_single_task,
cross_validate_by_player_single_task) specifically to avoid that.

Comparing this against conv2d_heads.py's tab performance tells you whether
the multi-task setup (training pitch and tab jointly) helps or hurts tab
prediction specifically -- multi-task learning can go either way: shared
representations can help (that's the whole reason pitch_head was kept as
an auxiliary task back when foundational_heads.py's two-head design was
discussed), or the two objectives can compete for the same limited
backbone capacity and hurt each other, especially at this model's small
parameter budget. Since tab is the actual product output (guitar tab is
string+fret, not note names), this comparison matters more than the
pitch-only one.

Reuses PrecomputedGuitarDataset (with its sliding-window tiling),
compute_class_weights, find_cqt_jams_pairs, and
cross_validate_by_player_single_task from foundational_heads.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundational_heads import find_cqt_jams_pairs, cross_validate_by_player_single_task

# ── 1. The Model ─────────────────────────────────────────

class Conv2dTabOnlyModel(nn.Module):
    """Same joint time-frequency 2D CNN backbone as Conv2dGuitarModel, but
    with only tab_head -- forward() returns just tab_probs, not a
    (pitch_probs, tab_probs) pair.
    """
    def __init__(self, input_bins=252, num_frets=23, temporal_hidden=256,
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

        self.tab_head = nn.Linear(context_dim, 6 * num_frets)
        self.num_frets = num_frets

    def forward(self, x):
        # x shape: (batch, 1, time_frames, input_bins) or (batch, time_frames, input_bins)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        batch_size, _, time_frames, _ = x.shape

        feat = self.conv(x)                       # (batch, 32, time_frames, 7)
        feat = feat.permute(0, 2, 1, 3).reshape(batch_size, time_frames, -1)
        latent = self.freq_proj(feat)

        context, _ = self.temporal(latent)
        context = self.context_dropout(context)

        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = F.softmax(tab_logits, dim=-1)

        return tab_probs

# ── 2. Main Execution ────────────────────────────────────────────────

def main() -> None:
    input_bins = 252
    sr = 22050
    hop_length = 512
    time_frames = round(4 * sr / hop_length)   # 4s sliding window
    stride = round(1 * sr / hop_length)        # 1s stride
    num_frets = 23
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

    def build_model():
        return Conv2dTabOnlyModel(
            input_bins=input_bins,
            num_frets=num_frets,
            temporal_hidden=temporal_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    cross_validate_by_player_single_task(
        model_fn=build_model,
        cqt_files=cqt_files,
        jams_files=jams_files,
        task="tab",
        time_frames=time_frames,
        stride=stride,
        batch_size=batch_size,
        epochs=100,
        lr=0.001,
        patience=10,
        device=device,
        checkpoint_dir="output",
        checkpoint_prefix="conv2d_tab_only_checkpoint",
        extra_checkpoint_fields={
            "input_bins": input_bins,
            "num_frets": num_frets,
            "temporal_hidden": temporal_hidden,
            "latent_dim": latent_dim,
            "dropout": dropout,
        },
    )

    print("Cross-validation job successful.")

if __name__ == "__main__":
    main()
