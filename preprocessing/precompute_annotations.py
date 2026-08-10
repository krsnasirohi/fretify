"""Precomputes per-frame pitch/tab/onset labels from GuitarSet's .jams
files and caches them to disk, so training scripts stop re-parsing JAMS
(JSON load + per-string note search) on every dataset access. That cost
used to be paid once per clip; after PrecomputedGuitarDataset started
windowing each clip into ~10 chunks, it was being paid ~10x per epoch per
clip instead.

One .npz per unique .jams file (not per audio variant -- all 4 recording
variants of a performance share the same annotation), keyed by the jams
filename stem.

Stores the FULL 88-key piano range (MIDI 21-108, Basic Pitch's own
"A0..C8" convention) rather than just this project's 49-class guitar range
(MIDI 40-88), so the same cache can serve every script:
  - foundational_heads.py / gru_heads.py slice out the 49-class window
    they need (MIDI 40-88 = index 19:68 into the 88-key array).
  - basic_pitch_finetune.py uses the full 88-key range directly -- it's
    exactly Basic Pitch's own note/onset output space.
  - basic_pitch_heads.py uses the same 49-class slice as
    foundational_heads.py.

Frame rate here is this project's own CQT rate (hop_length=512, sr=22050,
~23.2ms/frame), matching PrecomputedGuitarDataset's original live-JAMS
convention exactly (verified byte-for-byte against it) -- this is the
right rate to be exact about, since foundational_heads.py/gru_heads.py are
the scripts with the actual severe bottleneck this cache exists to fix
(windowing multiplies label rebuilding ~10x per epoch there).

basic_pitch_heads.py/basic_pitch_finetune.py run at Basic Pitch's own,
finer frame rate (hop=256 samples, ~11.6ms/frame, exactly half of this
cache's). They map each of their own frames' absolute time to its nearest
cache frame independently (round(frame_time / cache_hop)). This is NOT
bit-exact versus live-parsing at their native resolution -- upsampling a
coarser cache necessarily loses up to ~1 cache-frame (~23ms) of precision
right at note onset/offset boundaries (verified: ~92-99% exact per-frame
agreement, concentrated in short bursts at note transitions, not scattered
noise). That's an intentional tradeoff: exact bit-for-bit reproduction
for BOTH frame rates from one shared cache is mathematically impossible
(round(2x) != 2*round(x) in general), and these two scripts' actual
bottleneck is audio I/O / model inference, not label parsing -- so a small,
localized, sub-frame precision loss here is an acceptable trade for
avoiding a second, differently-configured cache.

Run once with: python precompute_annotations.py
"""
import glob
import os

import jams
import librosa
import numpy as np

HOP_LENGTH = 512         # this project's own CQT hop -- see module docstring
SR = 22050
BASE_MIDI = 21           # A0 -- Basic Pitch's own convention
NUM_KEYS = 88            # A0..C8
MUTE_CLASS = 22
OPEN_STRINGS = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4, low string -> high

JAMS_DIR = "data/guitarset/annotation"
CQT_DIR = "output/processed_cqt"
OUTPUT_DIR = "output/per_frame_annotations"

# Longest suffix first so "_hex_cln" isn't left partially stripped by "_hex".
SUFFIXES = ["_hex_cln", "_hex", "_mic", "_mix"]


def clip_stem(stem: str) -> str:
    """Strips a recording-variant suffix to get the shared jams-matching name."""
    for suf in SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def find_total_frames_per_jams() -> dict:
    """Longest matched CQT variant's frame count for each jams stem, so the
    cached arrays are long enough to cover every recording variant that
    shares that annotation (mix/mic/hex/hex_cln can differ by a frame or
    two), and exactly match the frame counts PrecomputedGuitarDataset
    itself reads from these same CQT files."""
    totals = {}
    for npy_path in glob.glob(os.path.join(CQT_DIR, "*.npy")):
        raw_stem = os.path.splitext(os.path.basename(npy_path))[0]
        stem = clip_stem(raw_stem)
        n_frames = np.load(npy_path, mmap_mode="r").shape[0]
        totals[stem] = max(totals.get(stem, 0), n_frames)
    return totals


def build_labels(jams_path: str, total_frames: int):
    """Frame-aligned (pitch, onset, tab) arrays for one full clip. Uses
    librosa.time_to_frames (floor-based), matching
    PrecomputedGuitarDataset's original live-JAMS-parsing convention
    exactly -- not round() -- so this cache is byte-for-byte identical to
    what foundational_heads.py/gru_heads.py computed before caching."""
    Y_pitch88 = np.zeros((total_frames, NUM_KEYS), dtype=np.float32)
    Y_onset88 = np.zeros((total_frames, NUM_KEYS), dtype=np.float32)
    Y_tab = np.full((total_frames, 6), MUTE_CLASS, dtype=np.int64)

    jam = jams.load(jams_path, validate=False)
    note_annotations = jam.annotations.search(namespace="note_midi")

    for string_idx, string_ann in enumerate(note_annotations):
        for note in string_ann:
            start_frame = max(0, librosa.time_to_frames(note.time, sr=SR, hop_length=HOP_LENGTH))
            end_frame = min(total_frames, librosa.time_to_frames(note.time + note.duration, sr=SR, hop_length=HOP_LENGTH))
            if end_frame <= start_frame:
                continue

            midi_pitch = int(round(note.value))

            key_idx = midi_pitch - BASE_MIDI
            if 0 <= key_idx < NUM_KEYS:
                Y_pitch88[start_frame:end_frame, key_idx] = 1.0
                Y_onset88[start_frame, key_idx] = 1.0

            fret = midi_pitch - OPEN_STRINGS[string_idx]
            if 0 <= fret <= 21:
                Y_tab[start_frame:end_frame, string_idx] = fret

    return Y_pitch88, Y_onset88, Y_tab


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jams_files = sorted(glob.glob(os.path.join(JAMS_DIR, "*.jams")))
    if not jams_files:
        raise ValueError(f"No .jams files found under {JAMS_DIR}")

    print("Scanning precomputed CQT files for per-clip frame counts...")
    totals = find_total_frames_per_jams()
    if not totals:
        raise ValueError(f"No .npy files found under {CQT_DIR} -- run cqt.py first.")

    n_saved, n_skipped = 0, 0
    for jams_path in jams_files:
        stem = os.path.splitext(os.path.basename(jams_path))[0]
        total_frames = totals.get(stem)
        if total_frames is None:
            print(f"  SKIP {stem}: no matching audio file found")
            n_skipped += 1
            continue

        Y_pitch88, Y_onset88, Y_tab = build_labels(jams_path, total_frames)
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.npz")
        np.savez(
            out_path,
            Y_pitch88=Y_pitch88,
            Y_onset88=Y_onset88,
            Y_tab=Y_tab,
            hop_length=HOP_LENGTH,
            sr=SR,
            base_midi=BASE_MIDI,
        )
        n_saved += 1
        if n_saved % 60 == 0:
            print(f"  ... {n_saved}/{len(jams_files)} done")

    print(f"Done. {n_saved} annotation caches saved to {OUTPUT_DIR}/ ({n_skipped} skipped).")


if __name__ == "__main__":
    main()
