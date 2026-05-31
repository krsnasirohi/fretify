import numpy as np
import soundfile as sf

def create_multi_note_test(filename="triple_test.wav"):
    sr = 22050
    # Frequencies for E2, A2, and D3
    frequencies = [82.41, 110.00, 146.83]
    note_duration = 0.8  # seconds per note
    silence_duration = 0.2  # seconds of silence between notes
    
    combined_audio = []

    for freq in frequencies:
        # Generate the note
        t = np.linspace(0, note_duration, int(sr * note_duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t) # 0.5 volume to avoid clipping
        
        # Generate the silence
        silence = np.zeros(int(sr * silence_duration))
        
        # Add them to our list
        combined_audio.extend(wave)
        combined_audio.extend(silence)

    # Convert to numpy array and save
    audio_data = np.array(combined_audio).astype(np.float32)
    sf.write(filename, audio_data, sr)
    print(f"✅ Created {filename} with 3 distinct notes.")

if __name__ == "__main__":
    create_multi_note_test()