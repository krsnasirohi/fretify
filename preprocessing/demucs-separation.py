import io
import demucs
from pathlib import Path
from demucs.api import Separator

# 1. Initialize the 6-stem model
# Use 'cuda' instead of 'cpu' if you have an Nvidia graphics card
separator = Separator(model="htdemucs_6s", device="cpu")

# 2. Define your input and output paths
input_file = Path("song.wav")
output_dir = Path("separated_stems")

print(f"Starting separation for {input_file.name}...")

# 3. Run the separation
# This returns a dictionary where keys are instrument names and values are audio tensors
origin, separated = separator.separate_audio_file(input_file)

# 4. Save the separated stems to your drive
# The save_audio function automatically creates the output directory if it doesn't exist
for stem, audio in separated.items():
    output_path = output_dir / f"{stem}.wav"
    separator.save_audio(audio, output_path, samplerate=separator.samplerate)
    print(f"Saved: {output_path}")

print("Separation complete!")
