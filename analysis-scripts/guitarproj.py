import librosa
import numpy as np

def identify_guitar_note(audio_path):
    try:
        # 1. Load the audio file
        # y = audio time series, sr = sampling rate
        y, sr = librosa.load(audio_path, sr=None)

        # 2. Clean the audio (Trim silence from the start/end)
        y, _ = librosa.effects.trim(y)
    
        # Skip the first 100ms to let the string stabilize
        y_stable = y[int(sr * 0.1):]

        # 3. Use the YIN algorithm for pitch tracking
        # YIN is specifically great for monophonic (single note) pitch detection
        f0 = librosa.yin(y_stable, fmin=librosa.note_to_hz('E2'), fmax=librosa.note_to_hz('E6'))

        # 4. Filter out any 'NaN' or zeros to get a clean average frequency
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) == 0:
            return "No pitch detected."

        average_hz = np.median(f0_clean)

        # 5. Convert frequency to musical note name
        detected_note = librosa.hz_to_note(average_hz)
        
        return {
            "Note": detected_note,
            "Frequency": round(average_hz, 2)
        }

    except Exception as e:
        return f"Error processing file: {e}"

# --- Execution ---
file_location = 'test.wav' # Replace with your file path
result = identify_guitar_note(file_location)

if isinstance(result, dict):
    print(f"🎸 Analysis Complete!")
    print(f"Detected Note: {result['Note']}")
    print(f"Frequency: {result['Frequency']} Hz")
else:
    print(result)