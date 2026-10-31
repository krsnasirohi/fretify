import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from scripts.data import FrameWindows, load_clips, TEST_PLAYERS
from scripts.greedy_midi import best_fingering, TUNING

AUDIO_DIR = Path("data/guitarset/audio/audio_hex-pickup_debleeded")
MIDI_DIR = Path("output/midi")   # Basic Pitch results are cached here so reruns are fast

RAW = "Raw network (p > 0.5)"
TRUE_NOTES = "Network + true notes"
BP_NOTES = "Network + Basic Pitch notes"


def precision_recall_f1(pred, true):
    tp = np.sum(pred * true)
    fp = np.sum(pred * (1 - true))
    fn = np.sum((1 - pred) * true)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(p), float(r), float(2 * p * r / max(p + r, 1e-9))


def fretboard_to_pitches(fb):
    """(6, 25) 0/1 grid -> list of MIDI notes."""
    strings, frets = np.nonzero(fb)
    return [TUNING[s] + f for s, f in zip(strings, frets)]


def basic_pitch_frames(stem, n_frames):
    """Basic Pitch's notes for one clip, as a list of pitches per frame."""
    from scripts.transcribe_midi import audio_to_midi, midi_to_frame_pitches  # slow import, only load if needed

    midi_path = MIDI_DIR / f"{stem}.mid"
    if not midi_path.exists():
        MIDI_DIR.mkdir(parents=True, exist_ok=True)
        audio_to_midi(AUDIO_DIR / f"{stem}.wav", midi_path)
    return midi_to_frame_pitches(midi_path, n_frames)


def evaluate(model, players, use_basic_pitch=True):
    """Score a model on these players' clips. Returns {method: (precision, recall, f1)}."""
    trues = []
    preds = {RAW: [], TRUE_NOTES: [], BP_NOTES: []}

    for cqt, y, stem in load_clips(players):
        p = model.predict(FrameWindows([(cqt, y, stem)], batch_size=512, shuffle=False), verbose=0)
        trues.append(y)
        preds[RAW].append((p > 0.5).astype(np.float32))
        # Paper's method: true pitches + network probabilities -> best playable fingering
        preds[TRUE_NOTES].append(np.stack([best_fingering(fretboard_to_pitches(y[t]), p[t]) for t in range(len(y))]))
        # Real-world version: pitches come from Basic Pitch instead of the annotations
        if use_basic_pitch:
            bp = basic_pitch_frames(stem, len(y))
            preds[BP_NOTES].append(np.stack([best_fingering(bp[t], p[t]) for t in range(len(y))]))
        print(f"  done {stem}")

    true = np.concatenate(trues)
    return {name: precision_recall_f1(np.concatenate(v), true) for name, v in preds.items() if v}


def print_results(results):
    for name, (p, r, f) in results.items():
        print(f"{name:30s} precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a trained model on the test player.")
    parser.add_argument("--model", default="checkpoints/pf_cnn.keras")
    parser.add_argument("--no-basic-pitch", action="store_true", help="skip the Basic Pitch line (faster)")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    print_results(evaluate(model, TEST_PLAYERS, use_basic_pitch=not args.no_basic_pitch))
