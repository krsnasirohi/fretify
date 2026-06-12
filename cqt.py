import jams
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import librosa

# 1. Define the directory path
folder_path = Path("data/guitarset/audio/audio_hex-pickup_debleeded")

# 2. Loop through all items in the folder and filter for files
for file_path in folder_path.iterdir():
    if file_path.suffix.lower() == '.wav':
        print(f"Processing WAV file: {file_path.name}")
        audio_path = file_path

        y, sr = librosa.load(audio_path, sr=None)
        cqt_spectrogram = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=36))

        # FIX 1: Uncommented this line so cqt_db actually exists for specshow
        cqt_db = librosa.amplitude_to_db(cqt_spectrogram, ref=np.max)

        plt.figure(figsize=(12, 6))

        # 3. Use specshow to plot the matrix with logarithmic and time scales
        librosa.display.specshow(
            cqt_db, 
            sr=sr, 
            hop_length=512,      
            x_axis='time',       
            y_axis='cqt_note',   
            cmap='magma'         
        )

        # 4. Add visual enhancements
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Transform (CQT) Spectrogram')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Musical Pitch')
        plt.tight_layout()

        # 5. Render the image
        output_dir_cqt = Path("output/cqt_figures")
        
        # FIX 2: Removed the extra 's' from output_dir_cqts
        output_dir_cqt.mkdir(parents=True, exist_ok=True) 
        
        clean_name_cqt = Path(file_path).stem
        # Define the final export path
        save_path_cqt = output_dir_cqt / f"{clean_name_cqt}.png"

        # Save the figure
        plt.savefig(save_path_cqt, dpi=300, bbox_inches="tight")
        
        # FIX 3: Updated the variable name to match save_path_cqt
        print(f"Saved plot to: {save_path_cqt}")
        
        # Important: Close the figure so you don't run out of memory during the loop
        plt.close()