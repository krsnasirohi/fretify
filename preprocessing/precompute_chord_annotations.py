"""Precomputes per-frame chord labels from GuitarSet's .jams files, as a
separate, additive cache alongside precompute_annotations.py's existing
pitch/tab/onset cache. Kept as its own directory specifically so it never
requires touching (or re-uploading, e.g. to a remote training pod) the
per_frame_annotations/ cache already in use.

Uses the plain/lead-sheet "chord" annotation (42 distinct chord classes in
this dataset), not the richer note-informed one (588 classes -- far too
sparse to learn from only ~360 recordings, most classes would have only a
handful of examples).

Chord annotations were verified to cover each clip essentially
gap-free (checked: zero gaps between consecutive segments across the
dataset; trailing uncovered time after the last segment is <=0.02s,
under one frame). A NO_CHORD placeholder class is still included for any
leftover unpainted frames, the same defensive pattern as tab's mute_class
-- cheap insurance even though it should basically never fire.

Same frame rate (hop_length=512, sr=22050) and same per-clip frame-count
derivation (from output/processed_cqt) as precompute_annotations.py, so
this cache is frame-aligned with the existing one.

Run once with: python precompute_chord_annotations.py
"""
import glob
import json
import os

import jams
import numpy as np

HOP_LENGTH = 512
SR = 22050

JAMS_DIR = "data/guitarset/annotation"
CQT_DIR = "output/processed_cqt"
OUTPUT_DIR = "output/chord_annotations"

# Longest suffix first so "_hex_cln" isn't left partially stripped by "_hex".
SUFFIXES = ["_hex_cln", "_hex", "_mic", "_mix"]


def clip_stem(stem: str) -> str:
    for suf in SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def find_total_frames_per_jams() -> dict:
    totals = {}
    for npy_path in glob.glob(os.path.join(CQT_DIR, "*.npy")):
        raw_stem = os.path.splitext(os.path.basename(npy_path))[0]
        stem = clip_stem(raw_stem)
        n_frames = np.load(npy_path, mmap_mode="r").shape[0]
        totals[stem] = max(totals.get(stem, 0), n_frames)
    return totals


def build_chord_vocab(jams_files) -> list:
    """Sorted, deterministic list of every distinct plain chord label
    across the dataset. A chord's class index is its position in this
    list; NO_CHORD is appended as the final class."""
    vocab = set()
    for jams_path in jams_files:
        jam = jams.load(jams_path, validate=False)
        plain_chord = jam.annotations.search(namespace="chord")[0]
        for o in plain_chord:
            vocab.add(o.value)
    return sorted(vocab) + ["NO_CHORD"]


def build_labels(jams_path: str, total_frames: int, chord_to_idx: dict, no_chord_idx: int):
    Y_chord = np.full(total_frames, no_chord_idx, dtype=np.int64)

    jam = jams.load(jams_path, validate=False)
    plain_chord = jam.annotations.search(namespace="chord")[0]

    for o in plain_chord:
        start_frame = max(0, int(round(o.time * SR / HOP_LENGTH)))
        end_frame = min(total_frames, int(round((o.time + o.duration) * SR / HOP_LENGTH)))
        if end_frame <= start_frame:
            continue
        Y_chord[start_frame:end_frame] = chord_to_idx[o.value]

    return Y_chord


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jams_files = sorted(glob.glob(os.path.join(JAMS_DIR, "*.jams")))
    if not jams_files:
        raise ValueError(f"No .jams files found under {JAMS_DIR}")

    print("Building chord vocabulary...")
    vocab = build_chord_vocab(jams_files)
    chord_to_idx = {c: i for i, c in enumerate(vocab)}
    no_chord_idx = chord_to_idx["NO_CHORD"]
    print(f"{len(vocab)} classes (including NO_CHORD): {vocab}")

    with open(os.path.join(OUTPUT_DIR, "_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

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

        Y_chord = build_labels(jams_path, total_frames, chord_to_idx, no_chord_idx)
        out_path = os.path.join(OUTPUT_DIR, f"{stem}.npz")
        np.savez(out_path, Y_chord=Y_chord, hop_length=HOP_LENGTH, sr=SR, num_classes=len(vocab))
        n_saved += 1
        if n_saved % 60 == 0:
            print(f"  ... {n_saved}/{len(jams_files)} done")

    print(f"Done. {n_saved} chord annotation caches saved to {OUTPUT_DIR}/ ({n_skipped} skipped).")


if __name__ == "__main__":
    main()
