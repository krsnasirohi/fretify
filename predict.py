"""Run a trained GuitarFoundationalModel on a single audio clip and
report/plot the predicted pitch activity and tablature.

Usage:
    python predict.py path/to/clip.wav
    python predict.py path/to/clip.wav --checkpoint output/model_checkpoint.pt
"""
import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from foundational_heads import GuitarFoundationalModel

# Must match the preprocessing in cqt.py / PrecomputedGuitarDataset exactly,
# since the model only knows how to interpret CQT frames built this way.
SR = 22050
HOP_LENGTH = 512
N_BINS = 252
BINS_PER_OCTAVE = 36
FMIN = librosa.note_to_hz("E2")

MIN_MIDI = 40  # matches PrecomputedGuitarDataset.min_midi
MUTE_CLASS = 22
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # low E2 -> high E4
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]


def load_model(checkpoint_path: str, device: torch.device) -> GuitarFoundationalModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GuitarFoundationalModel(
        input_bins=checkpoint["input_bins"],
        latent_dim=checkpoint["latent_dim"],
        num_pitches=checkpoint["num_pitches"],
        num_frets=checkpoint["num_frets"],
        temporal_hidden=checkpoint.get("temporal_hidden", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def clip_to_cqt(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=SR)
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=HOP_LENGTH, fmin=FMIN,
        n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE,
    ))
    return cqt.T  # (time_frames, 252)


@torch.no_grad()
def predict(model: GuitarFoundationalModel, cqt: np.ndarray, device: torch.device):
    # The model has no fixed sequence length -- it applies per-frame layers,
    # so a whole clip's CQT can be fed in as a single (1, time, 252) batch.
    x = torch.tensor(cqt, dtype=torch.float32).unsqueeze(0).to(device)
    pitch_probs, tab_probs = model(x)
    return pitch_probs.squeeze(0).cpu().numpy(), tab_probs.squeeze(0).cpu().numpy()


def print_tab(tab_probs: np.ndarray, stride: int = 10) -> None:
    """Coarse text tab: one column every `stride` frames (~0.23s at defaults)."""
    fret_idx = tab_probs.argmax(axis=-1)  # (time, 6)
    times = librosa.frames_to_time(np.arange(fret_idx.shape[0]), sr=SR, hop_length=HOP_LENGTH)
    columns = list(range(0, fret_idx.shape[0], stride))

    for string in reversed(range(6)):  # high string on top, guitar tab convention
        cells = [
            "--" if fret_idx[t, string] == MUTE_CLASS else f"{fret_idx[t, string]:2d}"
            for t in columns
        ]
        print(f"{STRING_NAMES[string]:>3} |" + "|".join(cells))

    print("     " + "|".join(f"{times[t]:4.1f}" for t in columns))


def plot_prediction(pitch_probs: np.ndarray, tab_probs: np.ndarray, out_path: Path) -> None:
    times = librosa.frames_to_time(np.arange(pitch_probs.shape[0]), sr=SR, hop_length=HOP_LENGTH)
    fret_idx = tab_probs.argmax(axis=-1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].imshow(
        pitch_probs.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
        extent=[times[0], times[-1], MIN_MIDI, MIN_MIDI + pitch_probs.shape[1]],
    )
    axes[0].set_ylabel("MIDI pitch")
    axes[0].set_title("Predicted pitch activity")

    for string in range(6):
        frets = fret_idx[:, string].astype(float)
        frets[fret_idx[:, string] == MUTE_CLASS] = np.nan
        axes[1].plot(times, frets, ".", label=STRING_NAMES[string], markersize=3)
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Fret")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Predicted tablature (fret per string, blank = muted)")
    axes[1].legend(loc="upper right", ncol=6, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction plot to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained model on one audio clip.")
    parser.add_argument("audio_path", type=str, help="Path to a .wav clip")
    parser.add_argument("--checkpoint", type=str, default="output/model_checkpoint.pt")
    parser.add_argument("--out", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    cqt = clip_to_cqt(args.audio_path)
    pitch_probs, tab_probs = predict(model, cqt, device)

    print_tab(tab_probs)

    out_path = Path(args.out) if args.out else Path("output/predictions") / f"{Path(args.audio_path).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_prediction(pitch_probs, tab_probs, out_path)


if __name__ == "__main__":
    main()
