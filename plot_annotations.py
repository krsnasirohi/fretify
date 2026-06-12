import jams
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import librosa

# sys.argv[0] is the script name; sys.argv[1] is the first argument
if len(sys.argv) > 1:
    filename = sys.argv[1]
    print(f"Reading file: {filename}")
else:
    print("Please provide a filename.")

# 1. Load the JAMS file
jam = jams.load(filename)

# 2. Filter for pitch contours (F0 data)
# GuitarSet contains 6 pitch_contour annotations, one for each string
pitch_annotations = jam.search(namespace='pitch_contour')

plt.figure(figsize=(12, 6))

# 3. Loop through each string and plot the time vs frequency data
for i, ann in enumerate(pitch_annotations):
    times = []
    frequencies = []
    
    for observation in ann.data:
        # Each observation contains time, duration, value (dict), and confidence
        times.append(observation.time)
        frequencies.append(observation.value['frequency'])
    
    # Filter out zeros (where no note is played on that string)
    times = np.array(times)
    frequencies = np.array(frequencies)
    mask = frequencies > 0
    
    # Plot this string's frequency over time
    plt.scatter(times[mask], frequencies[mask], s=2, label=f'String {i+1}')

# 4. Format the graph
plt.xlabel('Time (Seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('GuitarSet Pitch Contour (Frequency vs Time)')
plt.legend()
plt.grid(True)

# Extract just the file name without the path or extension
clean_name = Path(filename).stem

# Create the output directory safely
output_dir = Path("output/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Define the final export path
save_path = output_dir / f"{clean_name}.png"

# Save the figure
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {save_path}")


