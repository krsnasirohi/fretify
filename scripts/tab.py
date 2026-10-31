"""Turn a guitar recording into ASCII tab.

  1. Basic Pitch finds which notes are played and when they start.
  2. pf-cnn gives string/fret probabilities for every frame.
  3. best_fingering picks a playable fingering at each note start.

The tab shows the order of notes, not their exact rhythm.

Run from the project root:
  python -m scripts.tab path/to/audio.wav [--model checkpoints/pf_cnn.keras] [--out song.txt]
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

from scripts.data import FrameWindows
from scripts.greedy_midi import best_fingering, fold_into_range, TUNING
from scripts.transcribe_midi import FRAME_RATE, HOP, SR, run_basic_pitch

STRING_NAMES = ["E", "A", "D", "G", "B", "e"]  # low -> high, same order as the model's rows
CHORD_GAP = 0.05    # notes starting within 50 ms of each other count as one chord
LINE_WIDTH = 80


def compute_cqt(audio_path):
    """Same settings as preprocessing/cqt.py, plus the log1p used in training."""
    y, _ = librosa.load(audio_path, sr=SR)
    cqt = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("E2"),
                             n_bins=252, bins_per_octave=36))
    return np.log1p(cqt.T).astype(np.float32)   # (frames, 252)


def predict_fretboards(model, cqt):
    """(frames, 252) CQT -> (frames, 6, 25) string/fret probabilities."""
    dummy_labels = np.zeros((len(cqt), 6, 25), dtype=np.float32)  # FrameWindows expects labels; unused here
    return model.predict(FrameWindows([(cqt, dummy_labels, "")], batch_size=512, shuffle=False), verbose=0)


def group_onsets(note_events):
    """Group notes that start at (nearly) the same time: [(start_seconds, [pitches]), ...]."""
    events = []
    for start, _, pitch, *_ in sorted(note_events, key=lambda n: n[0]):
        if events and start - events[-1][0] < CHORD_GAP:
            events[-1][1].append(int(pitch))
        else:
            events.append((start, [int(pitch)]))
    return events


def choose_fingerings(events, note_events, p):
    """For each onset, return {string: fret} for the notes that start there."""
    chords = []
    for start, new_pitches in events:
        t = min(int(round(start * FRAME_RATE)) + 1, len(p) - 1)  # one frame after the onset, once the note has settled
        now = t / FRAME_RATE
        # Notes still ringing from earlier also occupy strings, so include them when choosing a fingering
        ringing = [int(pitch) for s, e, pitch, *_ in note_events if s <= now < e]
        fretboard = best_fingering(ringing + new_pitches, p[t])

        # Only write the notes that start here; ringing notes were written earlier
        new = set(fold_into_range(new_pitches))
        strings, frets = np.nonzero(fretboard)
        played = {int(s): int(f) for s, f in zip(strings, frets) if TUNING[s] + f in new}
        if played:
            chords.append(played)
    return chords


def render_tab(chords):
    """Lay out chords as ASCII tab (high e on top), wrapped to LINE_WIDTH."""
    columns = []
    for played in chords:
        width = max(len(str(f)) for f in played.values())
        columns.append([str(played[s]).ljust(width, "-") if s in played else "-" * width for s in range(6)])

    def block_lines(block):
        return [STRING_NAMES[s] + "|-" + "--".join(col[s] for col in block) + "-|" for s in reversed(range(6))]

    lines, block, used = [], [], 0
    for col in columns:
        if block and used + len(col[0]) + 2 > LINE_WIDTH - 4:
            lines += block_lines(block) + [""]
            block, used = [], 0
        block.append(col)
        used += len(col[0]) + 2
    if block:
        lines += block_lines(block)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Turn a guitar recording into ASCII tab.")
    parser.add_argument("audio")
    parser.add_argument("--model", default="checkpoints/pf_cnn.keras")
    parser.add_argument("--out", help="where to save the tab (default: output/tabs/<name>.txt)")
    args = parser.parse_args()

    audio = Path(args.audio)
    model = tf.keras.models.load_model(args.model)

    print("Finding notes with Basic Pitch...")
    _, note_events = run_basic_pitch(audio)
    print("Predicting string/fret probabilities...")
    p = predict_fretboards(model, compute_cqt(audio))

    tab = render_tab(choose_fingerings(group_onsets(note_events), note_events, p))

    out = Path(args.out) if args.out else Path("output/tabs") / f"{audio.stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tab + "\n")
    print(tab)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
