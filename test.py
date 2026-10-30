import numpy as np
import tensorflow as tf

from data import FrameWindows, load_clips, TEST_PLAYERS
from greedy_midi import best_fingering, TUNING


def precision_recall_f1(pred, true):
    tp = np.sum(pred * true)
    fp = np.sum(pred * (1 - true))
    fn = np.sum((1 - pred) * true)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def fretboard_to_pitches(fb):
    """(6, 25) 0/1 grid -> list of MIDI notes."""
    strings, frets = np.nonzero(fb)
    return [TUNING[s] + f for s, f in zip(strings, frets)]


model = tf.keras.models.load_model("checkpoints/pf_cnn.keras")
clips = load_clips(TEST_PLAYERS)

trues, raws, fingered = [], [], []
for cqt, y, stem in clips:
    p = model.predict(FrameWindows([(cqt, y, stem)], batch_size=512, shuffle=False), verbose=0)
    trues.append(y)
    raws.append((p > 0.5).astype(np.float32))
    # Paper's method: true pitches + network probabilities -> best playable fingering
    fingered.append(np.stack([best_fingering(fretboard_to_pitches(y[t]), p[t]) for t in range(len(y))]))
    print(f"  done {stem}")

true = np.concatenate(trues)
for name, pred in [("Raw network (p > 0.5)", raws), ("Network + best_fingering", fingered)]:
    p, r, f = precision_recall_f1(np.concatenate(pred), true)
    print(f"{name:28s} precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
