"""Fine-tunes Spotify's pretrained Basic Pitch model directly on GuitarSet
-- no custom heads, just Basic Pitch's own note/onset outputs updated
end-to-end. This is a starting point: a from-scratch training loop you can
extend (e.g. add a contour loss, swap in your own heads once this
overfits/plateaus), not a finished pipeline.

Unlike basic_pitch_heads.py (which freezes Basic Pitch and feeds its
outputs into this project's existing PyTorch pitch_head/tab_head), this
script leaves every Basic Pitch weight trainable, which is only possible
in TensorFlow -- the CoreML/ONNX exports Basic Pitch also ships are
inference-only and can't be backpropped through.

SETUP -- this needs a SEPARATE virtualenv from the project's main one:
tensorflow-macos<2.15.1 (the version basic-pitch pins) has no wheels for
Python 3.14, which the main .venv here runs. Use an older interpreter:

    python3.10 -m venv .venv-basicpitch
    source .venv-basicpitch/bin/activate
    pip install "basic-pitch[tf]" jams librosa

Run with:
    source .venv-basicpitch/bin/activate
    python basic_pitch_finetune.py
"""
import glob
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import tensorflow as tf

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch import constants as bp_constants

from preprocessing.data_splits import grouped_split, player_id_from_stem, print_and_verify_split

WINDOW_SAMPLES = bp_constants.AUDIO_N_SAMPLES   # 43844 (~1.988s @ 22050Hz)
SAMPLE_RATE = bp_constants.AUDIO_SAMPLE_RATE    # 22050
N_FRAMES = bp_constants.ANNOT_N_FRAMES          # 172
FRAME_HOP = bp_constants.FFT_HOP / SAMPLE_RATE  # seconds/frame
N_KEYS = bp_constants.N_FREQ_BINS_NOTES         # 88 (A0..C8, Basic Pitch's own convention)
BASE_MIDI = 21                                  # A0 -- matches FREQ_BINS_NOTES[0]

ANNOTATION_CACHE_DIR = "output/per_frame_annotations"

# ── 1. GuitarSet file pairing (debleeded hex-pickup variant only) ──

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

# ── 2. Windowing + frame-aligned note/onset targets ────────────────────

def iter_windows(pairs: List[Tuple[str, str]]):
    """Yields (audio_segment, jams_path, window_start_time_sec). A trailing
    partial window is dropped rather than padded, so every window is fully
    real audio and needs no mask."""
    for audio_path, jams_path in pairs:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        n_windows = len(y) // WINDOW_SAMPLES
        for w in range(n_windows):
            segment = y[w * WINDOW_SAMPLES : (w + 1) * WINDOW_SAMPLES]
            yield segment.astype(np.float32), jams_path, w * WINDOW_SAMPLES / SAMPLE_RATE


def build_note_onset_targets(jams_path: str, start_time: float, onset_frame_tolerance: int = 1):
    """Frame-aligned (172, 88) note-active and onset targets, in Basic
    Pitch's own semitone-per-bin convention (bin i = MIDI 21+i, A0..C8) --
    NOT this project's guitar-specific 49-class pitch space, since this
    fine-tunes Basic Pitch's own output heads directly.

    Reads from the precomputed per-frame cache (see
    precompute_annotations.py) instead of re-parsing the JAMS file live.
    The cache is stored at this project's own CQT frame rate (~23.2ms/
    frame); this function's own frame rate (FRAME_HOP, ~11.6ms/frame) is
    about half of that, so each of this function's frames maps to its
    nearest cache frame independently -- see precompute_annotations.py's
    docstring for why this isn't bit-exact vs. live parsing at this finer
    rate (a small, localized, sub-frame precision loss at note boundaries,
    not a correctness bug).

    Contour has no target here: GuitarSet's note_midi annotations don't
    carry the continuous pitch-bend detail contour's 264-bin resolution is
    meant to supervise. Its weights still get gradient updates (note/onset
    are computed downstream of the contour branch in the graph), just
    without a direct loss term -- a reasonable place to extend this later.
    """
    stem = os.path.splitext(os.path.basename(jams_path))[0]
    cache_path = os.path.join(ANNOTATION_CACHE_DIR, f"{stem}.npz")
    cached = np.load(cache_path)
    cache_hop = float(cached["hop_length"]) / float(cached["sr"])
    Y_pitch88 = cached["Y_pitch88"]
    Y_onset88 = cached["Y_onset88"]

    frame_times = start_time + np.arange(N_FRAMES) * FRAME_HOP
    cache_indices = np.round(frame_times / cache_hop).astype(int)
    valid = (cache_indices >= 0) & (cache_indices < Y_pitch88.shape[0])

    note_target = np.zeros((N_FRAMES, N_KEYS), dtype=np.float32)
    raw_onset = np.zeros((N_FRAMES, N_KEYS), dtype=np.float32)
    note_target[valid] = Y_pitch88[cache_indices[valid]]
    raw_onset[valid] = Y_onset88[cache_indices[valid]]

    # Dilate raw onset flags by the requested tolerance -- same semantics
    # as the original per-note onset marking.
    onset_target = np.zeros_like(raw_onset)
    onset_frames, onset_keys = np.nonzero(raw_onset)
    for f, k in zip(onset_frames, onset_keys):
        onset_target[f : min(N_FRAMES, f + 1 + onset_frame_tolerance), k] = 1.0

    return note_target, onset_target


def compute_pos_weights(pairs: List[Tuple[str, str]]) -> Tuple[float, float]:
    """Global scalar pos_weight per head (note-on vs note-off, onset vs
    not), same spirit as compute_class_weights in foundational_heads.py
    but simplified to one scalar instead of per-class, since most of
    Basic Pitch's 88 piano keys are never played on guitar at all."""
    pos = np.zeros(2)
    total_pixels = 0
    for _, jams_path, start_time in iter_windows(pairs):
        note_t, onset_t = build_note_onset_targets(jams_path, start_time)
        pos[0] += note_t.sum()
        pos[1] += onset_t.sum()
        total_pixels += N_FRAMES * N_KEYS
    neg = total_pixels - pos
    weights = np.clip(neg / np.maximum(pos, 1), 1.0, 50.0)
    return float(weights[0]), float(weights[1])


def make_dataset(pairs: List[Tuple[str, str]], batch_size: int, shuffle: bool) -> tf.data.Dataset:
    def gen():
        for segment, jams_path, start_time in iter_windows(pairs):
            note_t, onset_t = build_note_onset_targets(jams_path, start_time)
            yield segment[:, np.newaxis], note_t, onset_t

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(WINDOW_SAMPLES, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(N_FRAMES, N_KEYS), dtype=tf.float32),
            tf.TensorSpec(shape=(N_FRAMES, N_KEYS), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=256)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ── 3. Model: full Basic Pitch, note+onset outputs only ────────────────

def build_finetune_model() -> tf.keras.Model:
    base = tf.keras.models.load_model(str(ICASSP_2022_MODEL_PATH), compile=False)
    # A new functional view over the same layers/weights as `base` (not a
    # copy) -- dropping contour from the *loss* while still training
    # through it, since note/onset are computed downstream of it.
    return tf.keras.Model(
        inputs=base.input,
        outputs={"note": base.get_layer("note").output, "onset": base.get_layer("onset").output},
        name="basic_pitch_finetune",
    )

# ── 4. Loss / metrics / train loop ──────────────────────────────────────

def weighted_bce(y_true, y_pred, pos_weight, eps=1e-7):
    p = tf.clip_by_value(y_pred, eps, 1 - eps)
    loss = -(pos_weight * y_true * tf.math.log(p) + (1 - y_true) * tf.math.log(1 - p))
    return tf.reduce_mean(loss)


def compute_f1(y_true, y_pred, threshold=0.5, eps=1e-8):
    pred = tf.cast(y_pred > threshold, tf.float32)
    tp = tf.reduce_sum(pred * y_true)
    fp = tf.reduce_sum(pred * (1 - y_true))
    fn = tf.reduce_sum((1 - pred) * y_true)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(f1)


@tf.function
def train_step(model, optimizer, x, note_true, onset_true, note_pos_weight, onset_pos_weight):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        loss_note = weighted_bce(note_true, outputs["note"], note_pos_weight)
        loss_onset = weighted_bce(onset_true, outputs["onset"], onset_pos_weight)
        loss = loss_note + loss_onset
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, loss_note, loss_onset


def eval_step(model, x, note_true, onset_true, note_pos_weight, onset_pos_weight):
    outputs = model(x, training=False)
    loss_note = weighted_bce(note_true, outputs["note"], note_pos_weight)
    loss_onset = weighted_bce(onset_true, outputs["onset"], onset_pos_weight)
    loss = loss_note + loss_onset
    return loss, loss_note, loss_onset, outputs["note"], outputs["onset"]


def train_finetune(model, train_ds, val_ds, epochs, lr, note_pos_weight, onset_pos_weight, patience=5):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_loss_note = train_loss_onset = 0.0
        n_batches = 0
        for x, note_true, onset_true in train_ds:
            loss, loss_note, loss_onset = train_step(
                model, optimizer, x, note_true, onset_true, note_pos_weight, onset_pos_weight
            )
            train_loss += float(loss)
            train_loss_note += float(loss_note)
            train_loss_onset += float(loss_onset)
            n_batches += 1
        n_batches = max(n_batches, 1)
        train_loss /= n_batches
        train_loss_note /= n_batches
        train_loss_onset /= n_batches

        val_loss = val_loss_note = val_loss_onset = 0.0
        note_f1_sum = onset_f1_sum = 0.0
        n_val_batches = 0
        for x, note_true, onset_true in val_ds:
            loss, loss_note, loss_onset, note_pred, onset_pred = eval_step(
                model, x, note_true, onset_true, note_pos_weight, onset_pos_weight
            )
            val_loss += float(loss)
            val_loss_note += float(loss_note)
            val_loss_onset += float(loss_onset)
            note_f1_sum += compute_f1(note_true, note_pred)
            onset_f1_sum += compute_f1(onset_true, onset_pred)
            n_val_batches += 1
        n_val_batches = max(n_val_batches, 1)
        val_loss /= n_val_batches
        val_loss_note /= n_val_batches
        val_loss_onset /= n_val_batches
        note_f1 = note_f1_sum / n_val_batches
        onset_f1 = onset_f1_sum / n_val_batches

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} (note={train_loss_note:.4f} onset={train_loss_onset:.4f}) "
            f"val_loss={val_loss:.4f} (note={val_loss_note:.4f} onset={val_loss_onset:.4f}) | "
            f"val_note_f1={note_f1:.3f} val_onset_f1={onset_f1:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [w.numpy().copy() for w in model.trainable_variables]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_weights is not None:
        for var, val in zip(model.trainable_variables, best_weights):
            var.assign(val)
        print(f"\nLoaded best weights (val_loss={best_val_loss:.4f})")

    return model

# ── 5. Main Execution ────────────────────────────────────────────────

def main() -> None:
    batch_size = 16

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

    print("Scanning training labels for note/onset class weights...")
    note_pos_weight, onset_pos_weight = compute_pos_weights(train_pairs)
    print(f"note_pos_weight={note_pos_weight:.2f} onset_pos_weight={onset_pos_weight:.2f}")

    train_ds = make_dataset(train_pairs, batch_size, shuffle=True)
    val_ds = make_dataset(val_pairs, batch_size, shuffle=False)

    print("Loading pretrained Basic Pitch model for fine-tuning (all weights trainable)...")
    model = build_finetune_model()

    # A small LR relative to foundational_heads.py's 1e-3 -- this is
    # fine-tuning a pretrained model (which already saw GuitarSet as one of
    # its own training sets), not training from scratch, so a large LR
    # risks wrecking the pretrained weights before adapting them.
    model = train_finetune(
        model, train_ds, val_ds,
        epochs=100, lr=1e-4,
        note_pos_weight=note_pos_weight, onset_pos_weight=onset_pos_weight,
        patience=5,
    )

    checkpoint_dir = Path("output/basic_pitch_finetuned")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir), save_format="tf")
    print(f"Saved fine-tuned model to: {checkpoint_dir}")

if __name__ == "__main__":
    main()
