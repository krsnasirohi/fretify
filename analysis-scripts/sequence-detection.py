import librosa
import numpy as np
import os

def detect_note_sequence(file_path):
    # Check if file exists first
    if not os.path.exists(file_path):
        print(f"❌ Error: The file '{file_path}' was not found in this folder.")
        return []

    print(f"🔍 Loading {file_path}...")
    y, sr = librosa.load(file_path, sr=None)
    print(f"✅ Loaded. Sample Rate: {sr}, Duration: {len(y)/sr:.2f} seconds")

    # Extract Pitch
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'), sr=sr)
    
    # Extract Volume
    rms = librosa.feature.rms(y=y)[0]
    max_vol = np.max(rms)
    print(f"📊 Peak Volume detected: {max_vol:.4f} (Threshold is 0.02)")

    detected_notes = []
    last_stable_note = None
    stability_threshold = 7 
    current_note_count = 0
    candidate_note = None

    for i in range(len(f0)):
        freq = f0[i]
        vol = rms[i]

        if vol > 0.01 and not np.isnan(freq): # Lowered threshold slightly to 0.01
            current_note = librosa.hz_to_note(freq)

            if current_note == candidate_note:
                current_note_count += 1
            else:
                candidate_note = current_note
                current_note_count = 1

            if current_note_count >= stability_threshold:
                if candidate_note != last_stable_note:
                    print(f"🎶 Found Note: {candidate_note}")
                    detected_notes.append(candidate_note)
                    last_stable_note = candidate_note
        elif vol <= 0.01:
            last_stable_note = None
            candidate_note = None
            current_note_count = 0

    return detected_notes

if __name__ == "__main__":
    # MAKE SURE THIS MATCHES YOUR FILENAME
    filename = 'voice.wav' 
    sequence = detect_note_sequence(filename)
    
    if not sequence:
        print("Empty sequence: No notes were stable enough or loud enough to be detected.")
    else:
        print(f"\n🎸 FINAL SEQUENCE: {' -> '.join(sequence)}")