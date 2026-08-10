"""Feeds Spotify's pretrained Basic Pitch model into this project's existing
pitch_head / tab_head (imported from foundational_heads.py), instead of
training the custom CQT+MLP backbone in foundational_heads.py from scratch.

Basic Pitch already does CQT + harmonic-stacking signal processing similar
in spirit to this project's own backbone. Its official weights only load
under TensorFlow (the CoreML/ONNX exports it also ships are inference-only,
so this backbone is frozen -- see basic_pitch_finetune.py if you want the
other direction, fine-tuning Basic Pitch's own weights).

SETUP -- this needs a SEPARATE virtualenv from the project's main one:
tensorflow-macos<2.15.1 (the version basic-pitch pins) has no wheels for
Python 3.14, which the main .venv here runs. Use an older interpreter:

    python3.10 -m venv .venv-basicpitch
    source .venv-basicpitch/bin/activate
    pip install "basic-pitch[tf]" torch jams librosa

Run with:
    source .venv-basicpitch/bin/activate
    python basic_pitch_heads.py
"""
import glob
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch import constants as bp_constants

from foundational_heads import GuitarFoundationalModel, train_model
from preprocessing.data_splits import grouped_split, player_id_from_stem, print_and_verify_split

# ── 1. Frozen Basic Pitch feature extractor ──────────────────────────

class BasicPitchFeatureExtractor:
    """Wraps Spotify's pretrained Basic Pitch model as a frozen feature
    extractor. Rather than reaching into its internal conv layers (whose
    names/shapes aren't a stable public API), this uses its three public
    output posteriorgrams -- onset (88), note (88), contour (264) -- as a
    440-dim-per-frame feature vector. The model is tiny (~16.8K params,
    almost entirely CQT + harmonic-stacking signal processing rather than
    learned depth), so its final outputs carry essentially all its
    pretraining value anyway.
    """
    WINDOW_SAMPLES = bp_constants.AUDIO_N_SAMPLES   # 43844 (~1.988s @ 22050Hz)
    SAMPLE_RATE = bp_constants.AUDIO_SAMPLE_RATE    # 22050
    N_FRAMES = bp_constants.ANNOT_N_FRAMES          # 172
    FRAME_HOP = bp_constants.FFT_HOP / bp_constants.AUDIO_SAMPLE_RATE  # seconds/frame
    FEATURE_DIM = bp_constants.N_FREQ_BINS_NOTES * 2 + bp_constants.N_FREQ_BINS_CONTOURS  # 88+88+264=440

    def __init__(self):
        self.model = tf.keras.models.load_model(str(ICASSP_2022_MODEL_PATH), compile=False)
        self.model.trainable = False

    def extract_batch(self, windows: np.ndarray) -> np.ndarray:
        """windows: (n, WINDOW_SAMPLES) float32 -> (n, N_FRAMES, FEATURE_DIM) float32"""
        x = windows[..., np.newaxis].astype(np.float32)
        outputs = self.model(x, training=False)
        return np.concatenate(
            [outputs["onset"].numpy(), outputs["note"].numpy(), outputs["contour"].numpy()],
            axis=-1,
        )

# ── 2. GuitarSet file pairing (audio has 4 recording variants per .jams) ──

def find_audio_jams_pairs(
    audio_dir: str = "data/guitarset/audio",
    jams_dir: str = "data/guitarset/annotation",
) -> List[Tuple[str, str]]:
    """Pairs each debleeded hex-pickup recording (audio_hex-pickup_debleeded,
    filename suffix _hex_cln) with its .jams annotation -- one clean
    recording per performance, not all 4 mic/mix/hex/hex_cln variants."""
    jams_files = glob.glob(os.path.join(jams_dir, "*.jams"))
    jams_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in jams_files}

    suffix = "_hex_cln"
    pairs = []
    for audio_path in sorted(Path(audio_dir).rglob(f"*{suffix}.wav")):
        clean = audio_path.stem[: -len(suffix)]
        if clean in jams_dict:
            pairs.append((str(audio_path), jams_dict[clean]))
    return pairs

# ── 3. Precompute & cache frozen Basic Pitch features ─────────────────

def precompute_basic_pitch_features(
    pairs: List[Tuple[str, str]],
    extractor: BasicPitchFeatureExtractor,
    cache_dir: str,
    batch_size: int = 16,
) -> List[Tuple[str, str, float]]:
    """Runs every GuitarSet clip through frozen Basic Pitch once, caching
    each non-overlapping ~2s window's (172, 440) feature block to disk --
    mirrors how cqt.py precomputes CQT tensors once instead of recomputing
    them every epoch. A trailing partial window (shorter than WINDOW_SAMPLES)
    is dropped rather than padded, so every cached window is fully real
    audio and no frame_mask bookkeeping is needed downstream.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    window_samples = extractor.WINDOW_SAMPLES
    sr = extractor.SAMPLE_RATE
    window_duration = window_samples / sr

    windows: List[Tuple[str, str, float]] = []
    pending_meta: List[Tuple[Path, float, str]] = []
    pending_audio: List[np.ndarray] = []

    def flush():
        if not pending_audio:
            return
        feats = extractor.extract_batch(np.stack(pending_audio))
        for (out_path, start_time, jams_path), feat in zip(pending_meta, feats):
            np.save(out_path, feat)
            windows.append((str(out_path), jams_path, start_time))
        pending_meta.clear()
        pending_audio.clear()

    for audio_path, jams_path in pairs:
        stem = Path(audio_path).stem
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        n_windows = len(y) // window_samples
        for w in range(n_windows):
            out_path = cache_path / f"{stem}__w{w:03d}.npy"
            start_time = w * window_duration
            if out_path.exists():
                windows.append((str(out_path), jams_path, start_time))
                continue
            segment = y[w * window_samples : (w + 1) * window_samples]
            pending_meta.append((out_path, start_time, jams_path))
            pending_audio.append(segment)
            if len(pending_audio) >= batch_size:
                flush()
    flush()

    return windows

# ── 4. Dataset over cached feature windows ────────────────────────────

class BasicPitchGuitarDataset(Dataset):
    """Frame-aligned pitch/tab labels built the same way as
    PrecomputedGuitarDataset in foundational_heads.py (49 pitch classes
    MIDI 40-88, 6-string x 23-fret tab labels), so the reused
    pitch_head/tab_head see the label space they were designed for.
    """
    def __init__(self, windows: List[Tuple[str, str, float]], annotation_cache_dir: str = "output/per_frame_annotations"):
        self.windows = windows
        self.min_midi = 40   # Low E2
        self.max_midi = 88   # High E6 (49 classes)
        self.num_pitches = self.max_midi - self.min_midi + 1
        self.mute_class = 22
        self.num_frets = self.mute_class + 1
        self.open_strings = [40, 45, 50, 55, 59, 64]
        self.n_frames = BasicPitchFeatureExtractor.N_FRAMES
        self.frame_hop = BasicPitchFeatureExtractor.FRAME_HOP
        self.annotation_cache_dir = annotation_cache_dir

    def __len__(self):
        return len(self.windows)

    def _build_labels(self, jams_path: str, start_time: float):
        """Loads this window's pitch/tab labels from the precomputed cache
        (see precompute_annotations.py) instead of re-parsing the JAMS file
        live. The cache is stored at this project's own CQT frame rate
        (hop_length=512/sr=22050, ~23.2ms/frame); Basic Pitch's frame rate
        here (self.frame_hop, ~11.6ms/frame) is about half of that, so each
        Basic Pitch frame's own absolute time is mapped to its nearest
        cache frame independently -- a fixed per-window integer offset
        (e.g. start_frame + i//2) looks equivalent but isn't: it silently
        drifts by a frame whenever start_time isn't itself aligned to a
        cache-frame boundary, which is most of the time.
        """
        stem = os.path.splitext(os.path.basename(jams_path))[0]
        cache_path = os.path.join(self.annotation_cache_dir, f"{stem}.npz")
        cached = np.load(cache_path)

        base_midi = int(cached["base_midi"])
        cache_hop = float(cached["hop_length"]) / float(cached["sr"])
        lo = self.min_midi - base_midi
        hi = self.max_midi - base_midi + 1

        Y_pitch88 = cached["Y_pitch88"]
        Y_tab_full = cached["Y_tab"]

        frame_times = start_time + np.arange(self.n_frames) * self.frame_hop
        cache_indices = np.round(frame_times / cache_hop).astype(int)
        valid = (cache_indices >= 0) & (cache_indices < Y_pitch88.shape[0])

        Y_pitch = np.zeros((self.n_frames, self.num_pitches), dtype=np.float32)
        Y_tab = np.full((self.n_frames, 6), self.mute_class, dtype=np.int64)
        Y_pitch[valid] = Y_pitch88[cache_indices[valid], lo:hi]
        Y_tab[valid] = Y_tab_full[cache_indices[valid]]

        return Y_pitch, Y_tab

    def __getitem__(self, idx):
        feature_path, jams_path, start_time = self.windows[idx]
        features = np.load(feature_path)  # (172, 440)
        Y_pitch, Y_tab = self._build_labels(jams_path, start_time)
        frame_mask = np.ones(self.n_frames, dtype=np.float32)  # every cached window is a full, unpadded chunk

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(Y_pitch, dtype=torch.float32),
            torch.tensor(Y_tab, dtype=torch.long),
            torch.tensor(frame_mask, dtype=torch.float32),
        )


def compute_class_weights(dataset: BasicPitchGuitarDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same idea as compute_class_weights in foundational_heads.py, adapted
    to this dataset's windowed (feature_path, jams_path, start_time) index
    (foundational_heads.py's version is tied to its one-file-per-example
    PrecomputedGuitarDataset and can't be reused directly here).
    """
    pitch_pos = np.zeros(dataset.num_pitches, dtype=np.int64)
    pitch_total_frames = 0
    tab_counts = np.zeros(dataset.num_frets, dtype=np.int64)

    for _, jams_path, start_time in dataset.windows:
        Y_pitch, Y_tab = dataset._build_labels(jams_path, start_time)
        pitch_pos += Y_pitch.sum(axis=0).astype(np.int64)
        pitch_total_frames += dataset.n_frames
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

# ── 5. Model: frozen Basic Pitch features -> GRU -> existing heads ────

class BasicPitchBackboneModel(nn.Module):
    """A fresh temporal GRU (sized for Basic Pitch's 440-dim feature
    vector) feeding this project's existing pitch_head/tab_head, imported
    directly from GuitarFoundationalModel rather than recreated -- this is
    the "feed Basic Pitch into my existing heads" model.
    """
    def __init__(self, feature_dim=BasicPitchFeatureExtractor.FEATURE_DIM, num_pitches=49, num_frets=23, temporal_hidden=256):
        super().__init__()
        self.temporal = nn.GRU(
            input_size=feature_dim,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # heads_source's own backbone/temporal are discarded -- only its
        # pitch_head/tab_head submodules are reused, so context_dim must
        # match: temporal_hidden here has to equal heads_source's.
        heads_source = GuitarFoundationalModel(num_pitches=num_pitches, num_frets=num_frets, temporal_hidden=temporal_hidden)
        self.pitch_head = heads_source.pitch_head
        self.tab_head = heads_source.tab_head
        self.num_frets = num_frets

    def forward(self, x):
        # x shape: (batch, time_frames=172, feature_dim=440)
        batch_size, time_frames, _ = x.shape
        context, _ = self.temporal(x)

        pitch_logits = self.pitch_head(context)
        pitch_probs = torch.sigmoid(pitch_logits)

        tab_logits = self.tab_head(context)
        tab_logits = tab_logits.view(batch_size, time_frames, 6, self.num_frets)
        tab_probs = torch.softmax(tab_logits, dim=-1)

        return pitch_probs, tab_probs

# ── 6. Main Execution ──────────────────────────────────────────────────

def main() -> None:
    batch_size = 16
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = find_audio_jams_pairs()
    if not pairs:
        raise ValueError("No matched audio/.jams pairs found under data/guitarset/.")
    print(f"Found {len(pairs)} audio/.jams pairs (mix/mic/hex/hex_cln recordings all included).")

    train_pairs, val_pairs = grouped_split(
        pairs, key_fn=lambda pair: player_id_from_stem(Path(pair[0]).stem)
    )
    print_and_verify_split(
        train_pairs, val_pairs,
        key_fn=lambda pair: player_id_from_stem(Path(pair[0]).stem),
        jams_fn=lambda pair: pair[1],
    )

    print("Loading frozen Basic Pitch model...")
    extractor = BasicPitchFeatureExtractor()

    print("Extracting/caching Basic Pitch features (first run only; cached under output/basic_pitch_features/ afterward)...")
    train_windows = precompute_basic_pitch_features(train_pairs, extractor, "output/basic_pitch_features/train")
    val_windows = precompute_basic_pitch_features(val_pairs, extractor, "output/basic_pitch_features/val")
    window_secs = BasicPitchFeatureExtractor.WINDOW_SAMPLES / BasicPitchFeatureExtractor.SAMPLE_RATE
    print(f"{len(train_windows)} train windows, {len(val_windows)} val windows (~{window_secs:.2f}s each).")

    train_ds = BasicPitchGuitarDataset(train_windows)
    val_ds = BasicPitchGuitarDataset(val_windows)

    print("Scanning training labels to compute class weights...")
    pitch_pos_weight, tab_class_weight = compute_class_weights(train_ds)
    pitch_pos_weight = pitch_pos_weight.to(device)
    tab_class_weight = tab_class_weight.to(device)
    print(f"Pitch pos_weight range: [{pitch_pos_weight.min():.2f}, {pitch_pos_weight.max():.2f}]")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    temporal_hidden = 256
    model = BasicPitchBackboneModel(temporal_hidden=temporal_hidden).to(device)
    print("Built model layout (frozen Basic Pitch features -> GRU -> existing heads).")

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
        patience=5,
    )

    checkpoint_path = Path("output/basic_pitch_heads_checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_dim": BasicPitchFeatureExtractor.FEATURE_DIM,
        "num_pitches": 49,
        "num_frets": 23,
        "temporal_hidden": temporal_hidden,
    }, checkpoint_path)
    print(f"Saved model checkpoint to: {checkpoint_path}")

if __name__ == "__main__":
    main()
