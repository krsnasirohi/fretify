import jams
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import librosa

# Not recursive search for .wav files - will need to implement as audios are split into four folders inside parent
folder_path = Path("data/guitarset/audio/audio_hex-pickup_debleeded")
output_dir_cqt = Path("output/cqt_figures")
output_dir_tensors = Path("data/guitarset/processed_cqt") # NEW: Directory for ML tensors

output_dir_cqt.mkdir(parents=True, exist_ok=True) 
output_dir_tensors.mkdir(parents=True, exist_ok=True)

for file_path in folder_path.iterdir():
    if file_path.suffix.lower() == '.wav':
        print(f"Processing WAV file: {file_path.name}")
        
        y, sr = librosa.load(file_path, sr=22050)
        
        # FIX: Enforce 252 bins starting at E2 so it matches your PyTorch model
        cqt_spectrogram = np.abs(librosa.cqt(
            y, sr=sr, hop_length=512, fmin=librosa.note_to_hz('E2'), 
            n_bins=252, bins_per_octave=36
        ))

        # Transpose to (time_frames, frequency_bins)
        cqt_tensor = cqt_spectrogram.T 

        # --- 1. Save the ML Tensor ---
        clean_name = file_path.stem
        tensor_path = output_dir_tensors / f"{clean_name}.npy"
        np.save(tensor_path, cqt_tensor)
        print(f"Saved tensor to: {tensor_path}")

        # --- 2. Save the Visualization (Optional, keeps your old logic) ---
        cqt_db = librosa.amplitude_to_db(cqt_spectrogram, ref=np.max)
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(cqt_db, sr=sr, hop_length=512, x_axis='time', y_axis='cqt_note', cmap='magma')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Transform (CQT) Spectrogram')
        plt.tight_layout()
        plt.savefig(output_dir_cqt / f"{clean_name}.png", dpi=300, bbox_inches="tight")
        plt.close()