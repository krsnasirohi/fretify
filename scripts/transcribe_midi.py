from pathlib import Path

import numpy as np
import pretty_midi
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

# Must match preprocessing/cqt.py so MIDI frames line up with CQT frames
SR = 22050
HOP = 512
FRAME_RATE = SR / HOP  # ~43 frames per second

# Guitar range in standard tuning: open low E (E2) to fret 24 on high E (E6)
GUITAR_MIN_HZ = pretty_midi.note_number_to_hz(40)
GUITAR_MAX_HZ = pretty_midi.note_number_to_hz(88)


def run_basic_pitch(audio_path):
    """Run Basic Pitch on a recording.

    Returns (midi_data, note_events). Each note event is
    (start_seconds, end_seconds, midi_pitch, amplitude, pitch_bends).
    """
    _, midi_data, note_events = predict(
        str(audio_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,       # raise to drop weak/false note starts
        frame_threshold=0.3,       # raise to drop quiet sustained notes
        minimum_note_length=58,    # ms
        minimum_frequency=GUITAR_MIN_HZ,
        maximum_frequency=GUITAR_MAX_HZ,
    )
    return midi_data, note_events


def audio_to_midi(audio_path, midi_out_path):
    """Run Basic Pitch on a recording and save the transcribed notes as a .mid file."""
    midi_data, _ = run_basic_pitch(audio_path)
    midi_data.write(str(midi_out_path))
    return midi_data


def midi_to_frame_pitches(midi_path, n_frames=None):
    """Return one list of active MIDI pitches per CQT frame, e.g. [[40, 47, 52], [], ...]."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    roll = midi.get_piano_roll(fs=FRAME_RATE)  # (128 pitches, time frames)

    if n_frames is not None:  # pad or trim to match the CQT length
        if roll.shape[1] < n_frames:
            roll = np.pad(roll, ((0, 0), (0, n_frames - roll.shape[1])))
        roll = roll[:, :n_frames]

    return [np.nonzero(roll[:, t])[0].tolist() for t in range(roll.shape[1])]


if __name__ == "__main__":
    audio = Path("data/guitarset/audio/example.wav")  # change to your file
    midi_path = Path("output/midi") / f"{audio.stem}.mid"
    midi_path.parent.mkdir(parents=True, exist_ok=True)

    audio_to_midi(audio, midi_path)
    print(f"Saved MIDI to: {midi_path}")

    cqt = np.load(Path("output/processed_cqt") / f"{audio.stem}.npy")  # (frames, 252)
    frame_pitches = midi_to_frame_pitches(midi_path, n_frames=cqt.shape[0])
    print(f"{len(frame_pitches)} frames, first non-empty:",
          next((p for p in frame_pitches if p), None))
